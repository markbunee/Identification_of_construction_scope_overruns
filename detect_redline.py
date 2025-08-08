import cv2
import numpy as np
import math
import pyproj
from ultralytics import YOLO  # 导入YOLOv8

# 全局变量用于缓存坐标转换器
TRANSFORMER_CACHE = {}
YOLO_MODEL = None  # YOLOv8模型全局变量

def initialize_yolo_model(model_path=r'D:\ASUS\desk\Identification_of_construction_scope_overruns\detect\redline.pt'):
    """初始化YOLOv8模型"""
    global YOLO_MODEL
    if YOLO_MODEL is None:
        YOLO_MODEL = YOLO(model_path)
    return YOLO_MODEL

def get_utm_transformer(longitude):
    """
    根据经度自动确定合适的UTM投影区域
    """
    # 计算UTM区域编号 (1-60)
    utm_zone = int((longitude + 180) / 6) + 1
    
    # 检查缓存
    if utm_zone in TRANSFORMER_CACHE:
        return TRANSFORMER_CACHE[utm_zone]
    
    # 创建新的转换器
    wgs84 = pyproj.CRS("EPSG:4326")
    utm_crs = pyproj.CRS(f"EPSG:326{utm_zone}")  # 默认北半球
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    
    # 缓存转换器
    TRANSFORMER_CACHE[utm_zone] = transformer
    return transformer

def lonlat_to_xy(lon, lat, transformer):
    """经纬度转平面坐标（米）"""
    return transformer.transform(lon, lat)

def project_points_single_zone(boundary_points_lonlat, drone_lon, drone_lat, drone_altitude_m, 
                             focal_length_mm, image_width_px, image_height_px, hfov_degrees):
    """
    单个施工区域的投影计算
    """
    # 参数单位转换
    altitude_m = drone_altitude_m
    focal_length_m = focal_length_mm / 1000
    
    # 将视场角转换为弧度
    hfov_rad = math.radians(hfov_degrees)
    
    # 计算焦距的像素当量
    focal_length_pixels = (image_width_px / 2) / math.tan(hfov_rad / 2)
    
    # 计算图像中心点
    center_x = image_width_px / 2
    center_y = image_height_px / 2
    
    # 获取UTM转换器
    transformer = get_utm_transformer(drone_lon)
    
    # 将无人机位置转换为平面坐标（米）"
    drone_x, drone_y = lonlat_to_xy(drone_lon, drone_lat, transformer)
    
    # 转换每个施工边界点
    projected_points = []
    for point_lonlat in boundary_points_lonlat:
        # 将边界点转换为平面坐标（米）
        point_x, point_y = lonlat_to_xy(point_lonlat[0], point_lonlat[1], transformer)
        
        # 计算施工点与无人机的水平距离差
        dx = point_x - drone_x
        dy = point_y - drone_y
        
        # 计算透视投影坐标
        u = center_x + focal_length_pixels * (dx / altitude_m)
        v = center_y - focal_length_pixels * (dy / altitude_m)
        
        projected_points.append((u, v))
    
    return np.array(projected_points)

def cohen_sutherland_clip(x1, y1, x2, y2, left, right, bottom, top):
    """线段裁剪算法"""
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
    
    def compute_outcode(x, y):
        code = INSIDE
        if x < left: code |= LEFT
        elif x > right: code |= RIGHT
        if y < bottom: code |= BOTTOM
        elif y > top: code |= TOP
        return code
    
    outcode1 = compute_outcode(x1, y1)
    outcode2 = compute_outcode(x2, y2)
    accept = False
    
    while True:
        if not (outcode1 | outcode2):
            accept = True
            break
        elif outcode1 & outcode2:
            break
        else:
            outcode_out = outcode1 if outcode1 else outcode2
            x, y = 0, 0
            
            if outcode_out & TOP:
                x = x1 + (x2 - x1) * (top - y1) / (y2 - y1)
                y = top
            elif outcode_out & BOTTOM:
                x = x1 + (x2 - x1) * (bottom - y1) / (y2 - y1)
                y = bottom
            elif outcode_out & RIGHT:
                y = y1 + (y2 - y1) * (right - x1) / (x2 - x1)
                x = right
            elif outcode_out & LEFT:
                y = y1 + (y2 - y1) * (left - x1) / (x2 - x1)
                x = left
            
            if outcode_out == outcode1:
                x1, y1 = x, y
                outcode1 = compute_outcode(x, y)
            else:
                x2, y2 = x, y
                outcode2 = compute_outcode(x, y)
    
    if accept:
        return (int(x1), int(y1)), (int(x2), int(y2))
    return None, None

def draw_zone_boundary(image, projected_points, zone_id, color=(0, 255, 0), thickness=2):
    """在图像上绘制单个区域的边界和标识"""
    height, width = image.shape[:2]
    boundary = (0, width, 0, height)
    
    # 绘制边界线
    for i in range(len(projected_points)):
        start_idx = i
        end_idx = (i + 1) % len(projected_points)
        start_point = projected_points[start_idx]
        end_point = projected_points[end_idx]
        
        # 裁剪线段
        clipped_start, clipped_end = cohen_sutherland_clip(
            start_point[0], start_point[1], end_point[0], end_point[1], *boundary
        )
        
        # 绘制可见线段
        if clipped_start and clipped_end:
            cv2.line(image, clipped_start, clipped_end, color, thickness)
    
    # 添加区域标识
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Zone: {zone_id}"
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x = width - text_size[0] - 20
    text_y = 40  # 所有区域标识都在同一位置
    
    # 绘制背景
    cv2.rectangle(image, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (0, 0, 0), -1)
    
    # 绘制文本
    cv2.putText(image, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
    
    return image

def detect_objects(image, projected_points_list):
    """使用YOLOv8进行目标检测，只识别红线区域外的物体"""
    # 初始化YOLO模型
    model = initialize_yolo_model()

    # 执行目标检测
    results = model(image)

    # 创建一个新的图像用于绘制检测结果
    annotated_image = image.copy()

    # 检查每个检测结果是否在施工区域内
    for result in results:
        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 检查中心点是否在施工区域内
            inside_construction_zone = False
            for projected_points in projected_points_list:
                points = np.array(projected_points, dtype=np.int32)
                distance = cv2.pointPolygonTest(points, (center_x, center_y), False)
                if distance >= 0:  # 点在多边形内或边界上
                    inside_construction_zone = True
                    break

            # 如果检测对象在施工区域内，跳过不绘制
            if inside_construction_zone:
                continue

            # 绘制检测框和标签
            conf = box.conf[0].item()
            cls_id = int(box.cls[0])
            label = f"{result.names[cls_id]} {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 绘制标签
            cv2.putText(annotated_image, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated_image

def process_image_with_detection_and_projection(image, drone_gps, drone_altitude_m, focal_length_mm, 
                                               image_size, zones, boundaries):
    """
    主接口函数：执行目标检测和施工区域投影
    只识别红线划定区域外的物体
    """
    # 提取无人机位置
    drone_lat = drone_gps["lat"]
    drone_lon = drone_gps["lon"]
    
    # 获取图像尺寸
    img_height, img_width = image.shape[:2]
    target_width = image_size["width"]
    target_height = image_size["height"]
    
    # 调整焦距以适应实际图像尺寸
    if img_width != target_width or img_height != target_height:
        scale_x = img_width / target_width
        scale_y = img_height / target_height
        adjusted_focal_length = focal_length_mm * (scale_x + scale_y) / 2
    else:
        adjusted_focal_length = focal_length_mm
    
    # 计算所有区域的投影点
    projected_points_list = []
    for i, zone_id in enumerate(zones):
        boundary_points = boundaries[i]
        
        # 转换为经纬度列表格式
        boundary_lonlat = [[point["lon"], point["lat"]] for point in boundary_points]
        
        # 计算投影点
        projected_points = project_points_single_zone(
            boundary_lonlat,
            drone_lon, drone_lat, drone_altitude_m,
            adjusted_focal_length,
            img_width, img_height,
            84.0  # 默认水平视场角84度
        )
        projected_points_list.append(projected_points)
    
    # 进行目标检测，只识别红线区域外的物体
    detected_image = detect_objects(image, projected_points_list)
    
    # 创建施工区域图像副本
    result_image = detected_image.copy()
    
    # 绘制施工区域边界
    for i, (projected_points, zone_id) in enumerate(zip(projected_points_list, zones)):
        result_image = draw_zone_boundary(result_image, projected_points, zone_id)
    
    return result_image

def process_construction_zones(image, drone_gps, drone_altitude_m, focal_length_mm, 
                               image_size, zones, boundaries):
    """
    施工区域投影函数
    """
    # 提取无人机位置
    drone_lat = drone_gps["lat"]
    drone_lon = drone_gps["lon"]
    
    # 获取图像尺寸
    img_height, img_width = image.shape[:2]
    target_width = image_size["width"]
    target_height = image_size["height"]
    
    # 调整焦距以适应实际图像尺寸
    if img_width != target_width or img_height != target_height:
        scale_x = img_width / target_width
        scale_y = img_height / target_height
        adjusted_focal_length = focal_length_mm * (scale_x + scale_y) / 2
    else:
        adjusted_focal_length = focal_length_mm
    
    # 创建图像副本
    result_image = image.copy()
    
    # 处理每个区域
    for i, zone_id in enumerate(zones):
        boundary_points = boundaries[i]
        
        # 转换为经纬度列表格式
        boundary_lonlat = [[point["lon"], point["lat"]] for point in boundary_points]
        
        # 计算投影点
        projected_points = project_points_single_zone(
            boundary_lonlat,
            drone_lon, drone_lat, drone_altitude_m,
            adjusted_focal_length,
            img_width, img_height,
            84.0  # 默认水平视场角84度
        )
        
        # 在图像上绘制区域边界和标识
        result_image = draw_zone_boundary(result_image, projected_points, zone_id)
    
    return result_image

# ===============================================================
# 示例使用
# ===============================================================
if __name__ == "__main__":
    # 输入参数
    drone_gps = {"lat": 31.20999999999999, "lon": 121.44}
    drone_altitude_m = 1000.0
    focal_length_mm = 35.0
    image_size = {"width": 800, "height": 600}
    zones = ["zone-A", "zone-B"]
    boundaries = [
        # 区域1边界点
        [
            {"lat": 31.21, "lon": 121.44},
            {"lat": 31.2, "lon": 121.445},
            {"lat": 31.195, "lon": 121.495},
            {"lat": 31.22, "lon": 121.5},
            {"lat": 31.26, "lon": 121.49},
            {"lat": 31.255, "lon": 121.455},
            {"lat": 31.21, "lon": 121.44}
        ],
        # 区域2边界点
        [
            {"lat": 39.9035, "lon": 116.4090},
            {"lat": 39.9040, "lon": 116.4100},
            {"lat": 39.9030, "lon": 116.4105},
            {"lat": 39.9025, "lon": 116.4095}
        ]
    ]

    # 读取输入图像
    input_image = cv2.imread(r"C:\Users\ASUS\drone_frames_20250804_230200\drone_frame_0000.png")
    
    # 处理图像（目标检测+施工区域投影）
    result_image = process_image_with_detection_and_projection(
        image=input_image,
        drone_gps=drone_gps,
        drone_altitude_m=drone_altitude_m,
        focal_length_mm=focal_length_mm,
        image_size=image_size,
        zones=zones,
        boundaries=boundaries
    )
    
    # 保存结果
    cv2.imwrite("final_result.jpg", result_image)