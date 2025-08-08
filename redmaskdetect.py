import cv2
import numpy as np
import torch
import time
import os
from segment_anything import sam_model_registry, SamPredictor

ADAPTIVE_THRESHOLD = 0.2

class ConstructionSiteMonitor:
    def __init__(self,
                 sam_checkpoint=r"D:\ASUS\desk\施工红线\sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 device="cuda",
                 fine_tune_iterations=3):
        # 如果模型不存在直接抛错，省去下载耗时
        if not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError(sam_checkpoint)

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        self.device = device
        self.fine_tune_iterations = max(1, fine_tune_iterations)

        # 减少特征点数量以降低 ORB 计算量
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # —— 其余状态变量与原版保持一致 ——
        self.first_mask = None
        self.current_mask = None
        self.current_contour = None
        self.contour_history = []
        self.frame_count = 0
        self.last_update_time = 0
        self.update_interval = 10
        self.initialized = False
        self.fine_tuned = False
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_frame = None
        self.motion_history = []
        self.force_update_queued = False
        self.last_mask_area = 0
        self.lost_count = 0
        self.max_lost_frames = 5
        self.sam_time = 0
        self.orb_time = 0
        self.total_frames = 0
        self.key_frame = None
        self.key_points = None
        self.key_descriptors = None
        self.initial_features = None
        self.initial_mask = None
        self.initial_contour = None
        self.mask_path = None
        self.motion_still_threshold = 0.05   # 平均光流 < 0.05 视为静止
        self.still_frames = 0                # 连续静止帧计数器

    # ------------- 计算优化 -------------
    def _calculate_motion_factor(self, frame):
        """每3帧计算一次光流，返回 (motion_factor, is_still)"""
        if self.prev_frame is None or self.frame_count % 3 != 0:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return (self.motion_history[-1] if self.motion_history else 0.2), False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean() / 50
        self.motion_history.append(min(1.0, motion))
        if len(self.motion_history) > 5:
            self.motion_history.pop(0)
        self.prev_frame = gray
        avg_motion = np.mean(self.motion_history)
        is_still = avg_motion < self.motion_still_threshold
        return avg_motion, is_still
        

    def _optical_flow_tracking(self, current_frame):
        """光流跟踪：减少 maxLevel，降低迭代次数"""
        if self.prev_frame is None or len(self.tracking_points) < 8:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.tracking_points, None, **self.lk_params)
        good_new = new_pts[status == 1]
        good_old = self.tracking_points[status == 1]
        if len(good_new) > 4:
            H, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            if H is not None:
                self.current_contour = cv2.transform(self.current_contour.astype(np.float32), H).astype(np.int32)
                mask = np.zeros_like(self.current_mask)
                cv2.fillPoly(mask, [self.current_contour], 255)
                self.current_mask = mask
                self._update_contour()
        self.tracking_points = good_new.reshape(-1, 2)
        self.prev_frame = gray

    # ------------- 其余接口保持语义一致 -------------
    def initialize_with_mask(self, frame, mask_path):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(mask_path)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        self.first_mask = mask.astype(np.uint8)
        self.current_mask = self.first_mask.copy()
        self.initial_mask = self.first_mask.copy()
        self._update_contour()
        self.initial_contour = self.current_contour.copy() if self.current_contour is not None else None
        self.key_frame = frame.copy()
        self._extract_features(frame)
        self.initial_features = {
            "frame": frame.copy(),
            "key_points": self.key_points,
            "key_descriptors": self.key_descriptors,
            "mask": mask.copy()
        }
        if self.current_contour is not None:
            mask_pts = np.zeros_like(self.current_mask)
            cv2.drawContours(mask_pts, [self.current_contour], -1, 255, -1)
            pts = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                          maxCorners=100, qualityLevel=0.01, minDistance=10, mask=mask_pts)
            self.tracking_points = pts.reshape(-1, 2) if pts is not None else []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        self.last_update_time = time.time()
        self.initialized = True
        self.initial_mask_area = cv2.contourArea(self.current_contour) if self.current_contour is not None else 0
        self.min_mask_area = self.initial_mask_area * 0.7
        self.max_mask_area = self.initial_mask_area * 0.9

        self.last_mask_area = self.initial_mask_area

    def _extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.key_points, self.key_descriptors = self.orb.detectAndCompute(gray, None)

    def _update_contour(self):
        contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            self.current_contour = max(contours, key=cv2.contourArea)
            self.contour_history.append(self.current_contour.copy())
            if len(self.contour_history) > 3:           # 减少到 3 个历史轮廓
                self.contour_history.pop(0)
        else:
            self.current_contour = None

    def fine_tune_mask(self, frame):
        if not self.initialized:
            print("请先初始化系统")
            return
        contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("微调失败：无法从掩膜中提取轮廓")
            return
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        box = np.array([x, y, x + w, y + h])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_frame)
        best_mask = None
        best_score = 0
        for _ in range(self.fine_tune_iterations):
            M = cv2.moments(max_contour)
            cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] else (x + w // 2, y + h // 2)
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box[None, :],
                multimask_output=True
            )
            if masks is not None and len(masks) > 0:
                idx = np.argmax(scores)
                if scores[idx] > best_score:
                    best_mask = masks[idx]
                    best_score = scores[idx]
        if best_mask is not None:
            self.current_mask = (best_mask.astype(np.uint8)) * 255
            self._update_contour()
            print(f"掩膜微调完成，置信度: {best_score:.4f}")
            self.fine_tuned = True
            self.last_mask_area = cv2.contourArea(self.current_contour) if self.current_contour is not None else 0

    def reinitialize_mask(self, frame):
        print("跟踪丢失，尝试重新初始化...")
        if self.initial_features is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_pts, cur_desc = self.orb.detectAndCompute(gray, None)
        matches = self.matcher.match(self.initial_features["key_descriptors"], cur_desc) if cur_desc is not None else []
        if len(matches) > 10:
            src = np.float32([self.initial_features["key_points"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([cur_pts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                warped = cv2.warpPerspective(self.initial_mask.astype(np.float32), H,
                                             (frame.shape[1], frame.shape[0]))
                warped = (warped > 127).astype(np.uint8) * 255
                self.current_mask = warped
                self._update_contour()
                self.key_frame = frame.copy()
                self._extract_features(frame)
                self.predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.lost_count = 0
                return True
        return False

    def process_frame(self, frame):
        if not self.initialized:
            return None
        self.frame_count += 1
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time

        if not self.fine_tuned:
            self.fine_tune_mask(frame)

        motion_factor, is_still = self._calculate_motion_factor(frame)
        if is_still:
            self.still_frames += 1
        else:
            self.still_frames = 0
        adaptive_interval = max(1, int(self.update_interval * (1.0 - motion_factor * 0.8)))
        if motion_factor > ADAPTIVE_THRESHOLD and not self.force_update_queued:
            adaptive_interval = min(3, adaptive_interval)
            self.force_update_queued = True
        elif motion_factor < ADAPTIVE_THRESHOLD / 2:
            self.force_update_queued = False

        if self.current_contour is None:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                self.reinitialize_mask(frame)
        else:
            self.lost_count = 0

        if self.current_contour is not None:
            start_orb = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cur_pts, cur_desc = self.orb.detectAndCompute(gray, None)
            matches = self.matcher.match(self.key_descriptors, cur_desc) if cur_desc is not None else []
            if len(matches) > 10:
                src = np.float32([self.key_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([cur_pts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is not None:
                    self.current_contour = cv2.perspectiveTransform(
                        self.current_contour.astype(np.float32), H).astype(np.int32)
                    mask_canvas = np.zeros_like(self.current_mask)
                    cv2.fillPoly(mask_canvas, [self.current_contour], 255)
                    self.current_mask = mask_canvas
                    self._update_contour()
                else:
                    self._optical_flow_tracking(frame)
            else:
                self._optical_flow_tracking(frame)
            self.orb_time += time.time() - start_orb

        if time_since_last_update >= adaptive_interval and self.still_frames < 150:
            self._update_segmentation(frame)
            self.last_update_time = current_time
            self.key_frame = frame.copy()
            self._extract_features(frame)

        self.total_frames += 1
        return self.current_contour

    def _update_segmentation(self, frame):
        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_frame)

        if self.current_contour is None and self.initial_contour is None:
            return

        contour = self.current_contour if self.current_contour is not None else self.initial_contour
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(contour)
        pts = np.array([[cx, cy],
                        [x + w // 4, y + h // 4],
                        [x + 3 * w // 4, y + h // 4],
                        [x + w // 4, y + 3 * h // 4],
                        [x + 3 * w // 4, y + 3 * h // 4]])
        labels = np.array([1, 1, 1, 1, 1])

        masks, scores, _ = self.predictor.predict(
            point_coords=pts,
            point_labels=labels,
            box=np.array([x, y, x + w, y + h])[None, :],
            multimask_output=True
        )
        if masks is not None and len(masks) > 0:
            idx = np.argmax(scores)
            best_mask = masks[idx]
            area = np.sum(best_mask)
            if self.initial_mask_area > 0:
                if area < 0.7 * self.initial_mask_area or area > 1.0 * self.initial_mask_area:
                    best_mask = self.initial_mask
            best_mask = (best_mask.astype(np.uint8)) * 255
            kernel = np.ones((5, 5), np.uint8)
            best_mask = cv2.dilate(best_mask, kernel, iterations=1)
            self.current_mask = best_mask
            self._update_contour()
            print(f"分割更新完成，面积: {np.sum(best_mask > 0)} 置信度: {scores[idx]:.4f}")
        self.sam_time += time.time() - start_time

    def get_performance_stats(self):
        if self.total_frames == 0:
            return {}
        sam_updates = max(1, self.frame_count / self.update_interval)
        return {
            "total_frames": self.total_frames,
            "sam_time": self.sam_time,
            "avg_sam_time": self.sam_time / sam_updates * 1000,
            "orb_time": self.orb_time,
            "avg_orb_time": self.orb_time / max(1, self.total_frames) * 1000,
            "update_interval": self.update_interval
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    monitor = ConstructionSiteMonitor(device=device, fine_tune_iterations=500)

    video_path = r"D:\ASUS\desk\施工红线\32c34d20cd6adfcd2d5de0cd491d1810.mp4"
    mask_path = r"C:\Users\ASUS\frames\32c34d20cd6adfcd2d5de0cd491d1810_first_frame.jpg_mask.jpg"

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")
    monitor.initialize_with_mask(first_frame, mask_path)

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_monitoring.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

    frame_cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        contour = monitor.process_frame(frame)
        overlay = frame.copy()
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)
            for i, hc in enumerate(monitor.contour_history):
                cv2.drawContours(overlay, [hc], -1, (0, 0, int(255 * (0.3 + 0.7 * i / len(monitor.contour_history))), 2))
        out.write(overlay)
        cv2.imshow("Construction Site Monitoring", overlay)
        frame_cnt += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n===== 性能统计 =====")
    stats = monitor.get_performance_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")