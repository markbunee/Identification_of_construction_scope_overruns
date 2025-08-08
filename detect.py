import cv2
import numpy as np
import argparse
import os
import time
import collections
from ultralytics import YOLO
import shapely.geometry as geom
from shapely.prepared import prep

def load_mask(mask_path):
    """加载掩膜图像"""
    try:
        with open(mask_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
            mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            return mask
    except Exception as e:
        print(f"掩膜加载错误: {str(e)}")
        return None

def extract_boundary_points(mask):
    """从掩膜中提取边界点"""
    if mask is None:
        return None
    
    # 二值化处理
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未检测到轮廓，使用掩膜外接矩形")
        # 如果没有轮廓，使用掩膜的外接矩形
        coords = np.column_stack(np.where(binary_mask > 0))
        if len(coords) == 0:
            return None
        rect = cv2.boundingRect(coords)
        x, y, w, h = rect
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    
    # 找到最大轮廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 简化轮廓（保留更多点）
    epsilon = 0.001 * cv2.arcLength(main_contour, True)
    simplified_contour = cv2.approxPolyDP(main_contour, epsilon, True)
    
    return simplified_contour.squeeze(axis=1)

def track_boundary(prev_frame, current_frame, prev_boundary):
    """跟踪边界点"""
    if prev_boundary is None or len(prev_boundary) == 0:
        return None
    
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # 准备光流输入
    prev_pts = prev_boundary.astype(np.float32).reshape(-1, 1, 2)
    
    # 计算光流
    current_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, current_gray, prev_pts, None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    if current_pts is None:
        return None
    
    # 过滤成功的跟踪点
    status = status.ravel().astype(bool)
    tracked_points = current_pts.reshape(-1, 2)[status]
    
    return tracked_points

def align_mask_to_frame(base_image, frame, mask):
    """
    使用鲁棒的特征匹配将掩膜对齐到当前帧
    改进点：使用SIFT特征+FLANN匹配器，增加匹配筛选和几何验证
    """
    # 提高图像质量（增强特征）
    def enhance_features(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    # 增强特征
    base_enhanced = enhance_features(base_image)
    frame_enhanced = enhance_features(frame)
    
    # 转换为灰度图
    base_gray = cv2.cvtColor(base_enhanced, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
    
    # 初始化SIFT检测器（精度更高）
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=9)
    
    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(base_gray, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)
    
    if des1 is None or des2 is None:
        print("无法检测特征点，使用原始掩膜")
        return mask
    
    # 使用FLANN匹配器（更适合SIFT特征）
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=32)#high
    search_params = dict(checks=200)#high
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's比率测试筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:   #low
            good_matches.append(m)
    
    # 如果好的匹配数量不足
    if len(good_matches) < 10:  #high
        print(f"匹配点不足({len(good_matches)}), 使用原始掩膜")
        return mask
    
    # 准备点集
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 使用RANSAC计算单应性矩阵，增加迭代次数提高精度
    H, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=10000)
    
    if H is not None:
        # 计算inliers数量
        inliers = np.sum(mask_ransac)
        inlier_ratio = inliers / len(good_matches)
        print(f"匹配点: {inliers}/{len(good_matches)} inliers (比例: {inlier_ratio:.2f})")
        
        # 如果inliers比例太低(小于0.5)，认为匹配不可靠
        if inlier_ratio < 0.01:
            print(f"匹配质量不佳，inliers比例低 ({inlier_ratio:.2f}), 使用原始掩膜")
            return mask
        
        aligned_mask = cv2.warpPerspective(mask, H, (frame.shape[1], frame.shape[0]), 
                                      flags=cv2.INTER_NEAREST)
        
        # 保存匹配可视化用于调试
        matched_img = cv2.drawMatches(base_image, kp1, frame, kp2, good_matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("feature_matches.jpg", matched_img)
        print("特征匹配可视化已保存为'feature_matches.jpg'")
        
        return aligned_mask
    else:
        print("单应性矩阵计算失败，直接使用原始掩膜")
        return mask

class ObjectTracker:
    """目标跟踪器类，用于管理检测到的目标"""
    def __init__(self):
        # 存储目标信息的字典
        self.objects = {}
        # 自增ID计数器
        self.next_id = 1
        # 帧率（稍后设置）
        self.fps = 30
        # 目标停留时间阈值（单位：秒）
        self.stay_threshold = 30  # 30秒
        
    def set_fps(self, fps):
        """设置帧率"""
        self.fps = fps
        
    def update(self, detections, boundary, frame_count):
        """更新跟踪器状态"""
        updated_ids = []
        
        # 准备多边形区域
        poly = geom.Polygon(boundary)
        prepared_poly = prep(poly)
        
        # 处理当前检测结果
        for detection in detections:
            # 解析检测结果
            # [x1, y1, x2, y2, conf, class]
            x1, y1, x2, y2 = detection[:4].astype(int)
            conf = detection[4]
            class_id = int(detection[5])
            
            # 计算目标中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_point = geom.Point(center_x, center_y)
            
            # 检查目标是否在边界内
            if prepared_poly.contains(center_point):
                continue  # 跳过边界内的目标
            
            # 尝试匹配现有目标
            matched_id = None
            min_distance = float('inf')
            
            # 为每个目标查找最佳匹配
            for obj_id, obj_info in self.objects.items():
                # 只考虑最近一次检测的目标
                last_position = obj_info['positions'][-1]
                dist = np.sqrt((center_x - last_position[0]) ** 2 + (center_y - last_position[1]) ** 2)
                
                # 距离阈值，防止匹配过远的目标
                if dist < 100 and dist < min_distance:
                    min_distance = dist
                    matched_id = obj_id
            
            # 更新或创建目标
            if matched_id is not None:
                # 更新现有目标
                self.objects[matched_id]['positions'].append((center_x, center_y))
                self.objects[matched_id]['bboxes'].append((x1, y1, x2, y2))
                self.objects[matched_id]['last_update'] = frame_count
                updated_ids.append(matched_id)
            else:
                # 创建新目标
                obj_id = self.next_id
                self.next_id += 1
                self.objects[obj_id] = {
                    'first_frame': frame_count,
                    'last_update': frame_count,
                    'positions': collections.deque([(center_x, center_y)], maxlen=100),
                    'bboxes': collections.deque([(x1, y1, x2, y2)], maxlen=100),
                    'color': (0, 255, 255),  # 初始颜色为黄色
                    'class_id': class_id,
                    'confidence': conf
                }
                updated_ids.append(obj_id)
        
        # 处理未更新的目标
        for obj_id in list(self.objects.keys()):
            if obj_id not in updated_ids:
                # 如果目标超过15秒未更新，移除它
                if frame_count - self.objects[obj_id]['last_update'] > 3 * self.fps:
                    del self.objects[obj_id]
                else:
                    # 保持现有状态
                    updated_ids.append(obj_id)
        
        # 更新目标颜色状态（如果停留时间超过阈值）
        for obj_id in updated_ids:
            if obj_id in self.objects:
                obj = self.objects[obj_id]
                duration = (frame_count - obj['first_frame']) / self.fps
                if duration >= self.stay_threshold:
                    obj['color'] = (0, 0, 255)  # 红色
    
    def draw_objects(self, frame):
        """在帧上绘制跟踪的目标"""
        for obj_id, obj_info in self.objects.items():
            # 获取最新的边界框
            bbox = obj_info['bboxes'][-1]
            x1, y1, x2, y2 = bbox
            
            # 绘制边界框
            color = obj_info['color']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 计算目标在帧中停留的时间
            duration = (obj_info['last_update'] - obj_info['first_frame']) / self.fps
            text = f"ID:{obj_id} {duration:.1f}s"
            
            # 绘制文本
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='施工红线跟踪系统')
    parser.add_argument('--video', type=str, default=r"D:\ASUS\desk\Identification_of_construction_scope_overruns\vidio\DJI_20250710170338_0011_V.MP4", help='输入视频文件路径')
    parser.add_argument('--mask', type=str, default=r"C:\Users\ASUS\frames\DJI_20250710174836_0007_V_first_frame.jpg_mask.jpg", help='初始掩膜图像路径')
    parser.add_argument('--base_image', type=str, default=r"C:\Users\ASUS\frames\DJI_20250710174836_0007_V_first_frame.jpg", help='掩膜对应的原始图像路径')
    parser.add_argument('--model', type=str, default=r'D:\ASUS\desk\Identification_of_construction_scope_overruns\detect\redline.pt', help='YOLOv8模型路径')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频文件路径')
    parser.add_argument('--show', action='store_true', help='实时显示跟踪结果')
    args = parser.parse_args()

    print(f"视频路径: {args.video}")
    print(f"掩膜路径: {args.mask}")
    print(f"原始图像路径: {args.base_image}")
    print(f"YOLOv8模型: {args.model}")
    
    # 加载YOLOv8模型
    model = YOLO(args.model)
    print("YOLOv8模型加载成功!")
    
    # 加载掩膜
    mask = load_mask(args.mask)
    if mask is None:
        print("无法加载掩膜，退出程序")
        return
    
    # 加载原始图像
    base_image = cv2.imread(args.base_image)
    if base_image is None:
        print(f"无法加载原始图像: {args.base_image}")
        return
    
    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频文件: {args.video}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频尺寸: {width}x{height}, 帧率: {fps:.1f}fps, 总帧数: {total_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 读取第一帧视频
    ret, first_frame = cap.read()
    if not ret:
        print("无法读取视频第一帧，退出程序")
        return
    
    # 将掩膜对齐到第一帧
    aligned_mask = align_mask_to_frame(base_image, first_frame, mask)
    
    # 从对齐后的掩膜提取边界点
    boundary_points = extract_boundary_points(aligned_mask)
    if boundary_points is None or len(boundary_points) < 3:
        print("无法提取有效边界点，退出程序")
        return
    
    print(f"初始边界点数量: {len(boundary_points)}")
    
    # 保存对齐后的掩膜用于调试
    cv2.imwrite("aligned_mask.jpg", aligned_mask)
    print("对齐后的掩膜已保存为'aligned_mask.jpg'")
    
    # 创建目标跟踪器
    tracker = ObjectTracker()
    tracker.set_fps(fps)
    
    print("开始处理视频...")
    start_time = time.time()
    prev_frame = first_frame.copy()
    current_boundary = boundary_points
    
    # 第一帧处理
    frame_count = 1
    # 在第一帧运行目标检测
    results = model.track(first_frame, persist=True, verbose=False)
    # 获取检测结果
    detections = results[0].boxes.data.cpu().numpy()
    # 更新跟踪器
    tracker.update(detections, current_boundary, frame_count)
    
    # 绘制边界和目标
    draw_frame = first_frame.copy()
    cv2.polylines(draw_frame, [current_boundary.astype(int)], True, (0, 0, 255), 3)
    tracker.draw_objects(draw_frame)
    
    # 写入输出帧
    out.write(draw_frame)
    
    # 如果显示开启，显示第一帧结果
    if args.show:
        cv2.imshow("施工红线跟踪", draw_frame)
        cv2.waitKey(1)
    
    # 用于目标检测的帧计数器
    detect_interval = 1  # 每240帧进行一次目标检测
    last_detect_frame = frame_count
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"处理帧 {frame_count}/{total_frames} ({frame_count/total_frames:.1%})", end='\r')
        
        # 跟踪边界点
        tracked_points = track_boundary(prev_frame, frame, current_boundary)
        
        # 更新边界
        if tracked_points is not None and len(tracked_points) > 2:
            current_boundary = tracked_points
        else:
            print(f"帧 {frame_count}: 跟踪失败，使用上一帧边界")
        
        # 每隔一定帧数运行目标检测
        if frame_count - last_detect_frame >= detect_interval:
            results = model.track(frame, persist=True, verbose=False)
            # 获取检测结果
            detections = results[0].boxes.data.cpu().numpy()
            # 更新跟踪器
            tracker.update(detections, current_boundary, frame_count)
            last_detect_frame = frame_count
            print(f"\n帧 {frame_count}: 执行目标检测，发现 {len(detections)} 个目标")
        
        # 绘制红线（边界）
        draw_frame = frame.copy()
        cv2.polylines(draw_frame, [current_boundary.astype(int)], True, (0, 0, 255), 3)
        
        # 绘制目标
        tracker.draw_objects(draw_frame)
        
        # 更新前一帧
        prev_frame = frame.copy()
        
        # 写入输出帧
        out.write(draw_frame)
        
        # 实时显示
        if args.show:
            cv2.imshow("施工红线跟踪", draw_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 清理资源
    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    
    # 打印性能统计
    elapsed = time.time() - start_time
    print(f"\n处理完成! 输出视频已保存至: {args.output}")
    print(f"总帧数: {frame_count}, 处理时间: {elapsed:.1f}秒, FPS: {frame_count/elapsed:.1f}")

if __name__ == "__main__":
    main()