import cv2
import numpy as np
import torch
import time
import os
import urllib.request
from segment_anything import sam_model_registry, SamPredictor

ADAPTIVE_THRESHOLD = 0.5  # 运动阈值系数

class ConstructionSiteMonitor:
    def __init__(self, sam_checkpoint=r"D:\ASUS\desk\施工红线\sam_vit_h_4b8939.pth", 
                 model_type="vit_h", device="cuda", fine_tune_iterations=3):
        # 确保模型文件存在
        self._ensure_sam_model_exists(sam_checkpoint)
        
        # 加载SAM模型
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.device = device
        self.fine_tune_iterations = fine_tune_iterations  # 微调次数
        
        # ORB特征检测器（用于镜头跟踪）
        self.orb = cv2.ORB_create(nfeatures=10000)
        
        # 创建暴力匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 系统状态
        self.first_mask = None
        self.current_mask = None
        self.current_contour = None
        self.contour_history = []  # 存储轮廓历史
        self.frame_count = 0
        self.last_update_time = 0
        self.update_interval = 10  # 每10秒更新一次分割
        self.initialized = False  # 标记是否完成初始化
        self.fine_tuned = False  # 标记是否完成微调
        self.lk_params = dict(winSize=(15,15), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_frame = None
        self.prev_points = None
        self.camera_height = 50  # 默认无人机高度(米)
        self.tracking_points = []
        self.motion_history = []
        self.force_update_queued = False  # 强制更新标记
        self.last_mask_area = 0  # 记录上一次掩膜面积
        self.lost_count = 0  # 跟踪丢失计数器
        self.max_lost_frames = 5  # 最大允许丢失帧数
        
        # 性能统计
        self.sam_time = 0
        self.orb_time = 0
        self.total_frames = 0
        
        # 存储关键帧和特征点
        self.key_frame = None
        self.key_points = None
        self.key_descriptors = None
        
        # 存储初始特征
        self.initial_features = None
        self.initial_mask = None
        self.initial_contour = None  # 存储初始轮廓
    
    def _calculate_motion_factor(self, frame):
        """计算场景运动因子0.0(静止)~1.0(剧烈运动)"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.2
        
        # 计算光流运动量
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, 
            current_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        motion_magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        motion_factor = np.mean(motion_magnitude) / 50  # 标准化运动因子
        
        # 更新历史记录
        self.motion_history.append(motion_factor)
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)
            
        # 更新前一帧
        self.prev_frame = current_gray.copy()
        
        return min(1.0, np.mean(self.motion_history))

    def _optical_flow_tracking(self, current_frame):
        """当特征匹配失败时使用光流追踪"""
        if self.prev_frame is None or len(self.tracking_points) < 10:
            self.prev_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return
        
        # 光流追踪
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, current_gray, 
            self.tracking_points, None, **self.lk_params
        )
        
        # 更新有效点
        good_new = new_points[status == 1]
        good_old = self.tracking_points[status == 1]
        
        if len(good_new) > 4:
            # 计算相似变换
            H, _ = cv2.estimateAffinePartial2D(good_old, good_new)
            if H is not None:
                # 变换轮廓
                transformed_contour = cv2.transform(self.current_contour.astype(np.float32), H)
                
                # 更新当前轮廓
                self.current_contour = transformed_contour.astype(np.int32)
                
                # 更新掩膜
                new_mask = np.zeros_like(self.current_mask)
                cv2.fillPoly(new_mask, [self.current_contour], 255)
                self.current_mask = new_mask
                
                # 更新轮廓历史
                self._update_contour()
        
        # 更新追踪点
        self.tracking_points = good_new.reshape(-1, 2)
        self.prev_frame = current_gray.copy()

    def _ensure_sam_model_exists(self, model_path):
        """确保SAM模型文件存在，如果不存在则下载"""
        if not os.path.exists(model_path):
            print(f"模型文件 {model_path} 不存在，正在下载...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("模型下载完成")
            except Exception as e:
                raise RuntimeError(f"无法下载模型文件: {str(e)}")
    
    def initialize_with_mask(self, frame, mask_path):
        """
        使用PNG掩膜文件初始化系统
        :param frame: 第一帧图像 (H, W, 3)
        :param mask_path: PNG掩膜文件路径
        """
        # 加载PNG掩膜文件
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩膜文件不存在: {mask_path}")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩膜文件: {mask_path}")
        
        # 确保掩膜与帧尺寸匹配
        if mask.shape != frame.shape[:2]:
            print(f"调整掩膜尺寸: {mask.shape} → {frame.shape[:2]}")
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # 二值化掩膜并转换为uint8
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        
        # 保存第一帧掩膜
        self.first_mask = mask
        self.current_mask = self.first_mask.copy()
        self.initial_mask = mask.copy()  # 保存初始掩膜用于重新检测
        
        # 提取轮廓
        self._update_contour()
        self.initial_contour = self.current_contour.copy()  # 保存初始轮廓
        
        # 设置关键帧和特征点
        self.key_frame = frame.copy()
        self._extract_features(frame)
        
        # 保存初始特征
        self.initial_features = {
            "frame": frame.copy(),
            "key_points": self.key_points,
            "key_descriptors": self.key_descriptors,
            "mask": mask.copy()
        }
        
        # 初始化追踪点
        if self.current_contour is not None:
            # 在轮廓内均匀采样点
            mask_points = np.zeros_like(self.current_mask)
            cv2.drawContours(mask_points, [self.current_contour], -1, 255, -1)
            self.tracking_points = cv2.goodFeaturesToTrack(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=10,
                mask=mask_points
            )
            if self.tracking_points is not None:
                self.tracking_points = self.tracking_points.reshape(-1, 2)
        
        # 使用安全的图像设置方法
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_frame)
        
        print("系统初始化完成")
        self.last_update_time = time.time()
        self.initialized = True
        self.last_mask_area = cv2.contourArea(self.current_contour) if self.current_contour is not None else 0
        # 记录初始面积
        self.initial_area = cv2.contourArea(self.initial_contour) if self.initial_contour is not None else 0

        # 记录初始掩膜的四个角点（用于形状锁定）
        x, y, w, h = cv2.boundingRect(self.initial_contour)
        self.initial_corners = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])

    def _extract_features(self, frame):
        """提取图像特征点和描述符"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.orb.detectAndCompute(gray, None)
        
        # 添加更多特征点以提高稳定性
        if len(key_points) < 300:
            self.orb.setMaxFeatures(1000)
            key_points, descriptors = self.orb.detectAndCompute(gray, None)
        
        self.key_points = key_points
        self.key_descriptors = descriptors
        
    def _update_contour(self):
        """从当前掩膜更新轮廓"""
        contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 取面积最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            self.current_contour = max_contour
            self.contour_history.append(max_contour.copy())
            
            # 只保留最近的5个轮廓
            if len(self.contour_history) > 5:
                self.contour_history = self.contour_history[-5:]
        else:
            self.current_contour = None
    
    def fine_tune_mask(self, frame):
        """使用SAM微调初始掩膜"""
        if not self.initialized:
            print("请先初始化系统")
            return
        
        print(f"开始微调初始掩膜 ({self.fine_tune_iterations}次迭代)...")
        
        # 使用初始掩膜作为提示 - 使用边界框而不是掩膜输入
        # 从掩膜中提取边界框
        contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("微调失败：无法从掩膜中提取轮廓")
            return
            
        # 取最大轮廓的边界框
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        box = np.array([x, y, x+w, y+h])
        
        # 使用SAM进行微调
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_frame)
        
        # 多次微调以提高精度
        best_mask = None
        best_score = 0
        
        for i in range(self.fine_tune_iterations):
            # 添加点提示增强精度
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 随机生成点提示
                point_coords = []
                point_labels = []
                
                # 中心点
                point_coords.append([cx, cy])
                point_labels.append(1)
                
                # 随机点
                for _ in range(4):
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(0, min(w, h)/2)
                    px = int(cx + distance * np.cos(angle))
                    py = int(cy + distance * np.sin(angle))
                    point_coords.append([px, py])
                    point_labels.append(1 if np.random.rand() > 0.3 else 0)  # 70%正样本
                
                point_coords = np.array(point_coords)
                point_labels = np.array(point_labels)
            else:
                point_coords = None
                point_labels = None
            
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box[None, :],  # 增加一个批次维度
                mask_input=None,
                multimask_output=True
            )
            
            # 选择最佳掩膜
            if masks is not None and len(masks) > 0:
                best_idx = np.argmax(scores)
                if scores[best_idx] > best_score:
                    best_mask = masks[best_idx]
                    best_score = scores[best_idx]
        
        # 更新掩膜和轮廓
        if best_mask is not None:
            self.current_mask = best_mask.astype(np.uint8)
            self._update_contour()
            print(f"掩膜微调完成，置信度: {best_score:.4f}")
            self.fine_tuned = True
            self.last_mask_area = cv2.contourArea(self.current_contour) if self.current_contour is not None else 0
        else:
            print("微调失败，未检测到任何掩膜")
    
    def reinitialize_mask(self, frame):
        """当跟踪丢失时重新初始化掩膜"""
        print("检测到跟踪丢失，尝试重新初始化...")
        
        # 使用初始特征重新检测
        if self.initial_features is None:
            print("无法重新初始化：缺少初始特征")
            return False
        
        # 提取当前帧特征
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points, current_descriptors = self.orb.detectAndCompute(gray, None)
        
        # 匹配特征点
        matches = None
        if self.initial_features["key_descriptors"] is not None and current_descriptors is not None:
            matches = self.matcher.match(self.initial_features["key_descriptors"], current_descriptors)
        
        if matches and len(matches) > 10:
            # 提取匹配点的位置
            src_pts = np.float32([self.initial_features["key_points"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([current_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 使用RANSAC计算单应性矩阵
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # 应用变换到初始掩膜
                transformed_mask = cv2.warpPerspective(
                    self.initial_mask.astype(np.float32), H, (frame.shape[1], frame.shape[0]))
                transformed_mask = (transformed_mask > 127).astype(np.uint8) * 255
                
                # 更新当前掩膜
                self.current_mask = transformed_mask
                self._update_contour()
                
                # 更新关键帧和特征点
                self.key_frame = frame.copy()
                self._extract_features(frame)
                
                # 重置预测器
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(rgb_frame)
                
                print("重新初始化成功")
                self.lost_count = 0
                return True
            except cv2.error as e:
                print(f"单应性矩阵计算失败: {str(e)}")
        
        print("重新初始化失败")
        return False
    
    def process_frame(self, frame):
        """处理视频帧"""
        if not self.initialized:
            print("系统未初始化")
            return None
        
        self.frame_count += 1
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        
        # 如果尚未微调，先进行微调
        if not self.fine_tuned:
            self.fine_tune_mask(frame)
        
        # 计算运动因子
        motion_factor = self._calculate_motion_factor(frame)
        
        # 动态更新策略
        adaptive_interval = max(1, int(self.update_interval * (1.0 - motion_factor * 0.8)))
        
        # 运动检测强制更新
        if motion_factor > ADAPTIVE_THRESHOLD and not self.force_update_queued:
            adaptive_interval = min(3, adaptive_interval)  # 剧烈运动时更频繁更新
            self.force_update_queued = True
        elif motion_factor < ADAPTIVE_THRESHOLD / 2:
            self.force_update_queued = False
        
        # 检查是否跟踪丢失
        if self.current_contour is None:
            self.lost_count += 1
            if self.lost_count > self.max_lost_frames:
                if not self.reinitialize_mask(frame):
                    # 尝试完全重新初始化
                    print("尝试完全重新初始化...")
                    self.initialize_with_mask(frame, mask_path)  # 需要保存mask_path
                    self.lost_count = 0
        else:
            self.lost_count = 0
        
        # 处理轮廓更新
        if self.current_contour is not None:
            start_orb = time.time()
            
            # 计算当前帧的特征点
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_points, current_descriptors = self.orb.detectAndCompute(gray, None)
            
            # 匹配特征点
            matches = None
            if self.key_descriptors is not None and current_descriptors is not None:
                matches = self.matcher.match(self.key_descriptors, current_descriptors)
            
            if matches and len(matches) > 10:
                # 提取匹配点的位置
                src_pts = np.float32([self.key_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([current_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # 使用RANSAC计算单应性矩阵
                try:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    # 应用变换到轮廓
                    transformed_contour = cv2.perspectiveTransform(
                        self.current_contour.astype(np.float32), H
                    ).astype(np.int32)
                    
                    # 更新当前轮廓
                    self.current_contour = transformed_contour
                    
                    # 更新掩膜
                    new_mask = np.zeros_like(self.current_mask)
                    cv2.fillPoly(new_mask, [transformed_contour], 255)
                    self.current_mask = new_mask
                    
                    self._update_contour()
                except cv2.error as e:
                    print(f"单应性矩阵计算失败: {str(e)}")
                    # 尝试使用光流追踪
                    self._optical_flow_tracking(frame)
            else:
                # 特征点不足，使用光流追踪
                self._optical_flow_tracking(frame)
            
            self.orb_time += time.time() - start_orb
        
        # 检查是否需要更新分割
        if time_since_last_update >= adaptive_interval:
            print(f"更新分割 | 距离上次更新: {time_since_last_update:.1f}秒 | 运动因子: {motion_factor:.2f}")
            self._update_segmentation(frame)
            self.last_update_time = current_time
            # 更新关键帧
            self.key_frame = frame.copy()
            self._extract_features(frame)
        
        self.total_frames += 1
        if motion_factor < 0.05:
            print("镜头静止，跳过更新")
            return self.current_contour

    def _is_shape_stable(self, new_contour):
        """检查新轮廓是否形状稳定（基于四个角点）"""
        if new_contour is None:
            return False
        x, y, w, h = cv2.boundingRect(new_contour)
        new_corners = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
        diff = cv2.norm(self.initial_corners, new_corners, cv2.NORM_L2)
        return diff < max(w, h) * 0.3  # 允许轻微偏移

    def _is_area_valid(self, new_contour):
        """检查新轮廓面积是否在合理范围内"""
        if new_contour is None:
            return False
        area = cv2.contourArea(new_contour)
        return abs(area - self.initial_area) / self.initial_area < 0.3  # ±30%
    def _update_segmentation(self, frame):
        """更新分割结果 - 直接使用新的分割结果，不延续之前的"""
        start_time = time.time()
        old_mask = self.current_mask
        old_contour = self.current_contour
        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用SAM进行分割
        self.predictor.set_image(rgb_frame)
        
        # 使用初始轮廓作为提示（如果可用）
        point_coords = None
        point_labels = None
        
        if self.initial_contour is not None:
            M = cv2.moments(self.initial_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 添加多点提示
                point_coords = np.array([[cx, cy], 
                                        [cx-50, cy], 
                                        [cx+50, cy],
                                        [cx, cy-50],
                                        [cx, cy+50]])
                point_labels = np.array([1, 1, 1, 1, 1])  # 都是正样本提示
        
        # 执行分割
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=False
        )
        
        if masks is not None and len(masks) > 0:
            best_mask = masks[0].astype(np.uint8)
            contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                new_contour = max(contours, key=cv2.contourArea)
                if self._is_area_valid(new_contour) and self._is_shape_stable(new_contour):
                    self.current_mask = best_mask
                    self.current_contour = new_contour
                    self._update_contour()
                    print("分割更新完成（通过校验）")
                else:
                    print("分割失败：面积或形状变化过大，跳过更新")
        
        self.sam_time += time.time() - start_time
    
    def get_performance_stats(self):
        """获取性能统计"""
        if self.total_frames == 0:
            return {}
        
        sam_updates = max(1, self.frame_count / self.update_interval)
        orb_time_per_frame = self.orb_time / max(1, self.total_frames)
        
        return {
            "total_frames": self.total_frames,
            "sam_time": self.sam_time,
            "avg_sam_time": self.sam_time / sam_updates,
            "orb_time": self.orb_time,
            "avg_orb_time": orb_time_per_frame * 1000,
            "update_interval": self.update_interval
        }


if __name__ == "__main__":
    # 初始化监控系统
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    monitor = ConstructionSiteMonitor(device=device, fine_tune_iterations=500)
    
    # 创建视频流
    video_path = r"D:\ASUS\desk\施工红线\32c34d20cd6adfcd2d5de0cd491d1810.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    
    # 获取第一帧
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")
    
    # 从PNG文件加载掩膜
    mask_path = r"C:\Users\ASUS\frames\32c34d20cd6adfcd2d5de0cd491d1810_first_frame.jpg_mask.jpg"
    
    # 保存mask_path用于重新初始化
    global_mask_path = mask_path
    
    # 系统初始化
    monitor.initialize_with_mask(first_frame, mask_path)
    
    # 创建输出视频
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "output_monitoring.mp4"
    output_writer = cv2.VideoWriter(output_path, 
                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                  30, 
                                  (w, h))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧并获取轮廓
        contour = monitor.process_frame(frame)
        
        # 创建可视化结果 - 只绘制红色轮廓
        overlay = frame.copy()
        
        # 绘制红色轮廓（如果有历史记录，绘制所有历史轮廓形成轨迹）
        if contour is not None:
            # 当前轮廓
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)
            
            # 历史轮廓
            for i, hist_contour in enumerate(monitor.contour_history):
                alpha = 0.3 + (i / len(monitor.contour_history)) * 0.7
                color = (0, 0, int(255 * alpha))
                cv2.drawContours(overlay, [hist_contour], -1, color, 2)
        else:
            # 如果没有检测到轮廓，显示警告
            cv2.putText(overlay, "WARNING: No contour detected!", (w//2-200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # 添加时间信息和更新倒计时
        current_time = time.time()
        time_since_update = current_time - monitor.last_update_time
        time_to_update = max(0, monitor.update_interval - time_since_update)
        
        cv2.putText(overlay, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Next update in: {time_to_update:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加状态信息
        status = "Fine-tuned" if monitor.fine_tuned else "Initial"
        cv2.putText(overlay, f"Status: {status}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加运动因子信息
        motion_factor = monitor.motion_history[-1] if monitor.motion_history else 0.0
        cv2.putText(overlay, f"Motion: {motion_factor:.2f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加掩膜面积信息
        if monitor.current_contour is not None:
            area = cv2.contourArea(monitor.current_contour)
            cv2.putText(overlay, f"Area: {area:.0f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加丢失计数器
        if monitor.lost_count > 0:
            cv2.putText(overlay, f"Lost: {monitor.lost_count}/{monitor.max_lost_frames}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 写入输出视频
        output_writer.write(overlay)
        
        # 显示结果
        cv2.imshow("Construction Site Monitoring", overlay)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    # 获取性能统计
    stats = monitor.get_performance_stats()
    print("\n===== 性能统计 =====")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {avg_fps:.2f}")
    print(f"SAM总处理时间: {stats.get('sam_time', 0):.2f}秒")
    print(f"SAM平均处理时间: {stats.get('avg_sam_time', 0)*1000:.2f}ms")
    print(f"ORB总处理时间: {stats.get('orb_time', 0):.2f}秒")
    print(f"ORB平均处理时间: {stats.get('avg_orb_time', 0):.2f}ms/帧")
    print(f"更新间隔: {stats.get('update_interval', 0)}秒")
    
    cap.release()
    output_writer.release()
    cv2.destroyAllWindows()