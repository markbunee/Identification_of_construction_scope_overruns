import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, laplace
import matplotlib.cm as cm
import matplotlib.font_manager as fm
from skimage import exposure
import os
import imageio
from datetime import datetime
import json
from PIL import Image
import math
import pyproj
import warnings  # 添加警告处理

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置中文字体
font_path = fm.findfont(fm.FontProperties(family='SimHei'))
if font_path:
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    for font_name in ['Microsoft YaHei', 'SimSun', 'KaiTi']:
        if fm.findfont(fm.FontProperties(family=font_name)):
            plt.rcParams['font.sans-serif'] = [font_name]
            break

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100 
plt.rcParams['savefig.bbox'] = 'tight'

# 全局动画对象
anim = None

def on_close(event):
    """处理窗口关闭事件"""
    if anim is not None and hasattr(anim, 'event_source') and anim.event_source is not None:
        anim.event_source.stop()
    plt.close('all')

# ======================================================================
# 坐标转换函数（经纬度↔平面坐标系）
# ======================================================================
def create_coord_transformer():
    """创建UTM坐标转换器"""
    ref_lon, ref_lat = 121.47, 31.23  # 上海附近参考点
    utm_zone = int((ref_lon + 180) / 6) + 1
    is_north = ref_lat >= 0
    
    wgs84 = pyproj.CRS("EPSG:4326")
    utm_crs = pyproj.CRS(f"EPSG:326{utm_zone}") if is_north else pyproj.CRS(f"EPSG:327{utm_zone}")
    transformer = pyproj.Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    return transformer

def lonlat_to_xy(lon, lat, transformer):
    """经纬度转平面坐标（米）"""
    x, y = transformer.transform(lon, lat)
    return x / 1000.0, y / 1000.0  # 转换为公里

def xy_to_lonlat(x_km, y_km, transformer):
    """平面坐标转经纬度"""
    x_m, y_m = x_km * 1000.0, y_km * 1000.0
    return transformer.transform(x_m, y_m, direction=pyproj.enums.TransformDirection.INVERSE)

# 全局坐标转换器
COORD_TRANSFORMER = create_coord_transformer()

# ======================================================================
# 透视投影函数（与文档2一致）
# ======================================================================
def project_points(terrain_points, drone_lon, drone_lat, drone_altitude_m, 
                  focal_length_mm, image_width_px, image_height_px, hfov_degrees):
    """
    使用透视投影模型计算施工边界点在图像上的位置
    参数：
        terrain_points: 施工边界点列表，每个点为(lon, lat, z)经纬度坐标
        drone_lon, drone_lat: 无人机位置（经纬度）
        drone_altitude_m: 无人机高度（米）
        focal_length_mm: 相机焦距（毫米）
        image_width_px, image_height_px: 图像尺寸（像素）
        hfov_degrees: 水平视场角（度）
    
    返回：
        投影点在图像坐标系中的坐标列表（图像左上角为原点(0,0)）
    """
    # 将无人机位置转换为平面坐标（公里）
    drone_x, drone_y = lonlat_to_xy(drone_lon, drone_lat, COORD_TRANSFORMER)
    altitude_km = drone_altitude_m / 1000.0  # 米转公里
    
    # 参数单位转换
    altitude_m = altitude_km * 1000  # 公里转米
    focal_length_m = focal_length_mm / 1000  # 毫米转米
    
    # 将视场角转换为弧度
    hfov_rad = math.radians(hfov_degrees)
    
    # 计算焦距的像素当量
    focal_length_pixels = (image_width_px / 2) / math.tan(hfov_rad / 2)
    
    # 计算图像中心点
    center_x = image_width_px / 2
    center_y = image_height_px / 2
    
    # 转换每个施工边界点
    projected_points = []
    for point in terrain_points:
        # 将边界点转换为平面坐标（公里）
        point_x, point_y = lonlat_to_xy(point[0], point[1], COORD_TRANSFORMER)
        
        # 计算施工点与无人机的水平距离差
        dx_km = point_x - drone_x
        dy_km = point_y - drone_y
        
        # 转换为米（保持精度）
        dx_m = dx_km * 1000
        dy_m = dy_km * 1000
        
        # 计算透视投影坐标
        u = center_x + focal_length_pixels * (dx_m / altitude_m)
        v = center_y - focal_length_pixels * (dy_m / altitude_m)  # 图像y轴向下，取负
        
        projected_points.append((u, v))
    
    return np.array(projected_points)

# ======================================================================
# Cohen-Sutherland 线段裁剪算法（与文档2一致）
# ======================================================================
def cohen_sutherland_clip(x1, y1, x2, y2, left, right, bottom, top):
    """
    使用Cohen-Sutherland算法裁剪线段
    """
    # 定义区域编码
    INSIDE = 0
    LEFT = 1
    RIGHT = 2
    BOTTOM = 4
    TOP = 8
    
    def compute_outcode(x, y):
        code = INSIDE
        if x < left:
            code |= LEFT
        elif x > right:
            code |= RIGHT
        if y < bottom:  # 注意：图像坐标y轴向下
            code |= BOTTOM
        elif y > top:   # 注意：图像坐标y轴向下
            code |= TOP
        return code
    
    # 计算端点的区域编码
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
    else:
        return None, None

# ======================================================================
# 1. 三维地形生成模块
# ======================================================================
def generate_terrain(size=300, octaves=8, sigma=12, seed=42):
    np.random.seed(seed)
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for octave in range(octaves):
        noise = np.random.randn(*X.shape)
        Z += 1 / (2 ** octave) * noise
    
    Z = gaussian_filter(Z, sigma=sigma)
    Z += 0.5 * np.exp(-(X**2 + Y**2)/2)
    Z -= 0.3 * np.exp(-((X-3)**2 + (Y+2)**2)/3)
    
    vegetation = np.clip(0.3 * np.sin(2*X)**2 + 0.2 * np.cos(3*Y)**2 - 0.1*Z, 0, 1)
    
    buildings = np.zeros_like(Z)
    building_mask = (X > -1) & (X < 1) & (Y > -4) & (Y < -2)
    buildings[building_mask] = 0.8 + 0.1 * np.random.rand(*Z[building_mask].shape)
    
    return X, Y, Z, vegetation, buildings

# ======================================================================
# 2. 施工范围定义模块（使用经纬度）
# ======================================================================
def define_construction_area(X, Y, Z):
    """
    定义施工范围多边形（使用经纬度）
    返回:
        poly_points: 施工边界点 (lon, lat, z) 列表
    """
    # 施工范围中心点（真实经纬度）
    ref_lon, ref_lat = 121.47, 31.23  # 上海附近
    
    # 施工范围多边形顶点（经纬度坐标）
    vertices_lonlat = np.array([
        [ref_lon - 0.03, ref_lat - 0.02],  # 左下
        [ref_lon - 0.025, ref_lat - 0.03],  # 左中
        [ref_lon + 0.025, ref_lat - 0.035],  # 右下
        [ref_lon + 0.03, ref_lat - 0.01],  # 右中
        [ref_lon + 0.02, ref_lat + 0.03],  # 右上
        [ref_lon - 0.015, ref_lat + 0.025],  # 左上
        [ref_lon - 0.03, ref_lat - 0.02]   # 闭合多边形
    ])
    
    # 转换为平面坐标用于地形
    poly_points = []
    for lon, lat in vertices_lonlat:
        x, y = lonlat_to_xy(lon, lat, COORD_TRANSFORMER)
        dist = (X - x)**2 + (Y - y)**2
        nearest_idx = np.unravel_index(dist.argmin(), dist.shape)
        z_val = Z[nearest_idx]
        poly_points.append([lon, lat, z_val])
    
    return np.array(poly_points)

# ======================================================================
# 3. 无人机飞行轨迹生成模块（使用经纬度）
# ======================================================================
def generate_drone_path(poly_points, fixed_height_m=1000.0, num_points=150):
    """
    生成沿施工边界飞行的轨迹（固定高度）
    参数:
        poly_points: 施工边界点 (lon, lat, alt)
        fixed_height_m: 固定飞行高度（米）
        num_points: 轨迹点数
    返回:
        path_points: 无人机轨迹点列表 [(lon, lat, alt_m)]
    """
    # 将经纬度转换为平面坐标
    path_points_xy = []
    for point in poly_points:
        x, y = lonlat_to_xy(point[0], point[1], COORD_TRANSFORMER)
        path_points_xy.append([x, y])
    
    path_points_xy = np.array(path_points_xy)
    path_points_xy = np.vstack([path_points_xy, path_points_xy[0]])  # 闭合多边形
    
    # 在参数空间上插值生成平滑轨迹
    t = np.linspace(0, 1, len(path_points_xy))
    from scipy.interpolate import CubicSpline
    cs_x = CubicSpline(t, path_points_xy[:, 0])
    cs_y = CubicSpline(t, path_points_xy[:, 1])
    
    # 生成等间距的轨迹点（平面坐标）
    t_new = np.linspace(0, 1, num_points)
    x_path = cs_x(t_new)
    y_path = cs_y(t_new)
    
    # 转换回经纬度
    path_points = []
    for x, y in zip(x_path, y_path):
        lon, lat = xy_to_lonlat(x, y, COORD_TRANSFORMER)
        path_points.append([lon, lat, fixed_height_m])
    
    return np.array(path_points)

# ======================================================================
# 统一图片尺寸函数
# ======================================================================
def resize_to_target(image_path, target_size=(800, 600)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size, Image.LANCZOS)
        img.save(image_path)
    except Exception as e:
        print(f"调整图片尺寸时出错: {e}")

# ======================================================================
# 4. 地形可视化与无人机动画模块（使用一致参数）
# ======================================================================
def animate_drone_movement(X, Y, Z, vegetation, buildings, poly_points, drone_path):
    global anim  # 声明全局动画对象
    
    frame_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"drone_frames_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    records_dir = f"drone_records_{timestamp}"
    os.makedirs(records_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 14))
    fig.canvas.mpl_connect('close_event', on_close)
    
    # 创建3D地形图
    ax1 = fig.add_subplot(221, projection='3d')
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=cm.terrain, vert_exag=0.1, blend_mode='soft')
    
    surf = ax1.plot_surface(
        X, Y, Z, 
        facecolors=rgb,
        rstride=3, 
        cstride=3,
        alpha=0.9,
        linewidth=0,
        antialiased=True
    )
    
    # 获取平面坐标用于绘图
    poly_xy = np.array([lonlat_to_xy(p[0], p[1], COORD_TRANSFORMER) for p in poly_points])
    poly_x, poly_y = poly_xy[:, 0], poly_xy[:, 1]
    poly_z = poly_points[:, 2]
    
    ax1.plot(poly_x, poly_y, poly_z, 
             'r-', linewidth=3, markersize=8, marker='o', 
             markerfacecolor='yellow', markeredgecolor='red',
             label='施工红线')
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [list(zip(poly_x, poly_y, poly_z))]
    poly_collection = Poly3DCollection(verts, alpha=0.3, facecolor='red')
    ax1.add_collection3d(poly_collection)
    
    drone_point_3d, = ax1.plot([], [], [], 'bo', markersize=12, label='无人机位置')
    
    # 获取无人机平面坐标用于绘图
    drone_xy = np.array([lonlat_to_xy(p[0], p[1], COORD_TRANSFORMER) for p in drone_path])
    drone_x, drone_y = drone_xy[:, 0], drone_xy[:, 1]
    drone_z = [p[2] / 1000.0 for p in drone_path]  # 米转公里用于绘图
    
    ax1.plot(drone_x, drone_y, drone_z, 
             'b--', linewidth=1.5, alpha=0.7, label='飞行轨迹')
    
    ax1.set_zlim(np.min(Z)-0.5, np.max(Z)+0.5)
    ax1.view_init(elev=35, azim=-60)
    ax1.set_title('无人机飞行监控（3D视图）', fontsize=14)
    ax1.set_xlabel('东向坐标 (km)', fontsize=10)
    ax1.set_ylabel('北向坐标 (km)', fontsize=10)
    ax1.set_zlabel('海拔高度', fontsize=10)
    ax1.legend(loc='upper right')
    
    # 创建2D平面投影视图
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, Z, 20, cmap='terrain', alpha=0.7)
    plt.colorbar(contour, ax=ax2, label='海拔高度')

    veg_layer = ax2.imshow(vegetation, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                          cmap='Greens', alpha=0.4, origin='lower')
    
    building_coords = np.argwhere(buildings > 0.5)
    if len(building_coords) > 0:
        for i, j in building_coords[::50]:
            ax2.add_patch(Rectangle((X[0,j]-0.1, Y[i,0]-0.1), 0.2, 0.2, 
                                  color='red', alpha=0.7))
            
    poly_2d = Polygon(poly_xy, closed=True)
    patch = PatchCollection([poly_2d], alpha=0.5, color='red')
    ax2.add_collection(patch)
    
    ax2.plot(poly_x, poly_y, 
             'ro-', linewidth=2, markersize=5)
    ax2.text(np.mean(poly_x), np.mean(poly_y), 
             '施工区域', fontsize=12, color='darkred',
             ha='center', va='center', weight='bold')
    
    drone_point_2d, = ax2.plot([], [], 'bo', markersize=8, label='无人机位置')
    ax2.plot(drone_x, drone_y, 'b--', linewidth=1.5, alpha=0.7, label='飞行轨迹')
    
    height_text = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, fontsize=12,
                          bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_title('无人机飞行监控（2D视图）', fontsize=14)
    ax2.set_xlabel('东向坐标 (km)', fontsize=10)
    ax2.set_ylabel('北向坐标 (km)', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='lower right')
    
    # 创建俯拍视图
    ax3 = fig.add_subplot(223)
    ax3.set_title('无人机俯拍画面（模拟摄像头）', fontsize=14, color='#333333')
    drone_view = ax3.imshow(np.zeros((100,100)), cmap='terrain', 
                          vmin=Z.min(), vmax=Z.max(), 
                          extent=[0, 1, 0, 1], origin='lower')
    
    ax3.grid(False)
    ax3.add_patch(Rectangle((0.85, 0.05), 0.1, 0.015, color='white'))
    ax3.text(0.9, 0.09, '100m', color='white', ha='center', fontsize=8)
    
    drone_point_top, = ax3.plot([], [], 'bo', markersize=10)
    height_text_top = ax3.text(0.05, 0.95, '', transform=ax3.transAxes, color='white', 
                             bbox=dict(facecolor='black', alpha=0.5))
    boundary_line, = ax3.plot([], [], 'r-', linewidth=2, alpha=0.7)
    
    # 创建局部放大视图
    ax4 = fig.add_subplot(224)
    ax4.set_title('地形与植被分析', fontsize=14)
    
    terrain_rgb = np.zeros((*Z.shape, 3))
    terrain_rgb[..., 0] = np.clip((Z - Z.min()) / (Z.max() - Z.min()) * 0.7, 0, 1)
    terrain_rgb[..., 1] = vegetation * 0.8
    terrain_rgb[..., 2] = 0.2
    
    building_mask = buildings > 0.5
    terrain_rgb[building_mask] = [1, 0, 0]
    
    local_view = ax4.imshow(terrain_rgb, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                          alpha=0.9, origin='lower')
    
    ax4.plot(poly_x, poly_y, 'y-', linewidth=2.5)
    drone_point_local, = ax4.plot([], [], 'bo', markersize=10)
    
    # 动画更新函数
    def update(frame):
        nonlocal frame_counter
        drone_lon, drone_lat, drone_altitude_m = drone_path[frame]
        drone_x, drone_y = drone_xy[frame]
        drone_z = drone_altitude_m / 1000.0  # 米转公里用于绘图
        
        drone_point_3d.set_data([drone_x], [drone_y])
        drone_point_3d.set_3d_properties([drone_z])
        
        drone_point_2d.set_data([drone_x], [drone_y])
        
        height_text.set_text(f'无人机高度: {drone_altitude_m:.0f} m\n'
                            f'轨迹点: {frame+1}/{len(drone_path)}')
        
        ax1.view_init(elev=35, azim=-60 + frame*0.5)
        
        view_size = 0.8
        x_min = max(X.min(), drone_x - view_size/2)
        x_max = min(X.max(), drone_x + view_size/2)
        y_min = max(Y.min(), drone_y - view_size/2)
        y_max = min(Y.max(), drone_y + view_size/2)
        
        # 检查边界是否有效
        if x_min >= x_max or y_min >= y_max:
            # 如果边界无效，使用默认值
            x_min, x_max = drone_x - view_size/2, drone_x + view_size/2
            y_min, y_max = drone_y - view_size/2, drone_y + view_size/2
        
        # 获取地形数据
        try:
            x_mask = (X[0] >= x_min) & (X[0] <= x_max)
            y_mask = (Y[:,0] >= y_min) & (Y[:,0] <= y_max)
            
            # 确保有数据点
            if np.any(x_mask) and np.any(y_mask):
                terrain_block = Z[y_mask, :][:, x_mask]
            else:
                terrain_block = np.array([[0]])  # 使用默认值
        except:
            terrain_block = np.array([[0]])  # 使用默认值
        
        # 处理地形块
        if terrain_block.size > 0:
            try:
                if np.max(terrain_block) != np.min(terrain_block):
                    terrain_norm = (terrain_block - np.min(terrain_block)) / (np.max(terrain_block) - np.min(terrain_block))
                else:
                    terrain_norm = np.zeros_like(terrain_block)
                    
                if terrain_norm.size > 0:
                    try:
                        terrain_enhanced = exposure.equalize_hist(terrain_norm)
                        terrain_enhanced += 0.2 * laplace(terrain_norm)
                        
                        h, w = terrain_enhanced.shape
                        if h > 0 and w > 0:
                            y_grid, x_grid = np.ogrid[-h//2:h//2, -w//2:w//2]
                            mask = np.exp(-(x_grid**2 + y_grid**2) / (2*(h/4)**2))
                            terrain_enhanced = terrain_enhanced * mask + terrain_enhanced * (1-mask)*0.6
                            
                            drone_view.set_data(terrain_enhanced)
                            drone_view.set_extent([x_min, x_max, y_min, y_max])
                            drone_view.set_clim(Z.min(), Z.max())
                    except:
                        pass
            except:
                pass
        
        drone_point_top.set_data([drone_x], [drone_y])
        height_text_top.set_text(f'高度: {drone_altitude_m:.0f} m')
        boundary_line.set_data(poly_x, poly_y)
        
        ax4.set_xlim(drone_x - 1.5, drone_x + 1.5)
        ax4.set_ylim(drone_y - 1.5, drone_y + 1.5)
        drone_point_local.set_data([drone_x], [drone_y])
        
        frame_filename = os.path.join(output_dir, f"drone_frame_{frame_counter:04d}.png")
        record_filename = os.path.join(records_dir, f"frame_{frame_counter:04d}_record.json")
        
        fig_drone = plt.figure(figsize=(8, 6), frameon=False)
        ax_drone = fig_drone.add_axes([0, 0, 1, 1])
        ax_drone.axis('off')
        
        # 使用文档2一致的透视投影参数
        focal_length_mm = 35.0
        image_width_px = 800
        image_height_px = 600
        hfov_degrees = 84.0
        
        # 绘制地形
        if terrain_enhanced.size > 0:
            ax_drone.imshow(terrain_enhanced, cmap='terrain', 
                            vmin=Z.min(), vmax=Z.max(), 
                            extent=[x_min, x_max, y_min, y_max], origin='lower')
            
            # 绘制投影点
            try:
                projected_points = project_points(
                    poly_points, 
                    drone_lon, drone_lat, drone_altitude_m,
                    focal_length_mm, image_width_px, image_height_px, hfov_degrees
                )
                
                height, width = image_height_px, image_width_px
                boundary = (0, width, 0, height)
                
                # 绘制所有线段
                for i in range(len(projected_points)):
                    start_idx = i
                    end_idx = (i + 1) % len(projected_points)
                    start_point = projected_points[start_idx]
                    end_point = projected_points[end_idx]
                    
                    clipped_start, clipped_end = cohen_sutherland_clip(
                        start_point[0], start_point[1], 
                        end_point[0], end_point[1],
                        *boundary
                    )
                    
                    if clipped_start and clipped_end:
                        u1, v1 = clipped_start
                        u2, v2 = clipped_end
                        
                        geo_x1 = x_min + (u1 / width) * (x_max - x_min)
                        geo_y1 = y_min + ((height - v1) / height) * (y_max - y_min)
                        geo_x2 = x_min + (u2 / width) * (x_max - x_min)
                        geo_y2 = y_min + ((height - v2) / height) * (y_max - y_min)
                        
                        ax_drone.plot([geo_x1, geo_x2], [geo_y1, geo_y2], 'r-', linewidth=2)
            except Exception as e:
                print(f"绘制投影点时出错: {e}")
        
        # 添加文本信息
        ax_drone.text(0.9, 0.05, '100m', color='white', 
                    ha='center', va='center', fontsize=8,
                    transform=ax_drone.transAxes)
        
        ax_drone.text(0.05, 0.95, f'高度: {drone_altitude_m:.0f} m', 
                    color='white', fontsize=8,
                    transform=ax_drone.transAxes)
        
        ax_drone.text(0.05, 0.05, f'坐标: ({drone_lon:.6f}, {drone_lat:.6f})', 
                    color='white', fontsize=8,
                    transform=ax_drone.transAxes)
        
        # 添加区域标识
        ax_drone.text(0.95, 0.95, 'zone-1', color='white', fontsize=10,
                    ha='right', va='top', weight='bold',
                    bbox=dict(facecolor='red', alpha=0.5),
                    transform=ax_drone.transAxes)
        
        plt.savefig(frame_filename, dpi=120, bbox_inches='tight', pad_inches=0)
        plt.close(fig_drone)
        resize_to_target(frame_filename, target_size=(800, 600))
        
        # 创建记录文件（与文档2格式一致）
        frame_record = {
            "frame_number": frame_counter,
            "timestamp": datetime.now().isoformat(),
            "drone_position": {
                "lat": float(drone_lat),
                "lon": float(drone_lon),
                "altitude_m": float(drone_altitude_m)
            },
            "view_area": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max)
            },
            "image_path": os.path.abspath(frame_filename),
            # 施工边界点使用经纬度格式
            "boundary_points": [{"lat": float(p[1]), "lon": float(p[0])} for p in poly_points],
            "camera_parameters": {
                "focal_length_mm": focal_length_mm,
                "image_width": image_width_px,
                "image_height": image_height_px,
                "hfov_degrees": hfov_degrees
            },
            "zone_id": "zone-1"  # 区域标识
        }
        
        with open(record_filename, 'w') as f:
            json.dump(frame_record, f, indent=2)
        
        csv_path = os.path.join(records_dir, "flight_summary.csv")
        if frame_counter == 0:
            with open(csv_path, 'w') as f:
                f.write("frame,timestamp,lon,lat,altitude_m,x_min,x_max,y_min,y_max,image_path,zone_id\n")
        
        with open(csv_path, 'a') as f:
            f.write(f"{frame_counter},{frame_record['timestamp']},{drone_lon:.6f},{drone_lat:.6f},{drone_altitude_m:.1f},"
                    f"{x_min:.4f},{x_max:.4f},{y_min:.4f},{y_max:.4f},{frame_filename},zone-1\n")
        
        frame_counter += 1
        
        return (drone_point_3d, drone_point_2d, height_text, 
                drone_point_top, drone_point_local, height_text_top, 
                drone_view, local_view, boundary_line)
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(drone_path),
        interval=50,
        blit=True
    )
    
    plt.show()
    
    create_kml_trajectory(records_dir, drone_path, construction_area)
    create_gif_from_frames(output_dir)
    
    return anim

# ======================================================================
# 从保存的帧创建GIF动画
# ======================================================================
def create_gif_from_frames(frame_dir):
    try:
        images = []
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
        
        for filename in frame_files:
            filepath = os.path.join(frame_dir, filename)
            images.append(imageio.imread(filepath))
        
        if images:
            gif_path = os.path.join(frame_dir, "drone_flyby.gif")
            imageio.mimsave(gif_path, images, duration=0.1)
            print(f"已创建GIF动画: {gif_path}")
    except Exception as e:
        print(f"创建GIF时出错: {e}")

# ======================================================================
# 创建飞行轨迹KML文件（使用经纬度）
# ======================================================================
def create_kml_trajectory(records_dir, drone_path, poly_points):
    try:
        kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>无人机飞行轨迹</name>
<description>无人机沿施工边界的飞行路径</description>
<Style id="yellowLine">
    <LineStyle>
        <color>7f00ffff</color>
        <width>4</width>
    </LineStyle>
</Style>
<Placemark>
    <name>飞行路径</name>
    <description>无人机巡检路线</description>
    <styleUrl>#yellowLine</styleUrl>
    <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
"""
        for point in drone_path:
            kml_content += f"{point[0]},{point[1]},{point[2]} "
        
        kml_content += """
        </coordinates>
    </LineString>
</Placemark>
<Placemark>
    <name>施工区域</name>
    <description>施工边界范围</description>
    <Style>
        <LineStyle>
            <color>ff0000ff</color>
            <width>4</width>
        </LineStyle>
        <PolyStyle>
            <color>7f0000ff</color>
        </PolyStyle>
    </Style>
    <Polygon>
        <outerBoundaryIs>
            <LinearRing>
                <coordinates>
"""
        # 添加施工边界点
        for point in poly_points:
            kml_content += f"{point[0]},{point[1]},{point[2]} "
        # 闭合多边形
        kml_content += f"{poly_points[0][0]},{poly_points[0][1]},{poly_points[0][2]}"
        
        kml_content += """
                </coordinates>
            </LinearRing>
        </outerBoundaryIs>
    </Polygon>
</Placemark>
</Document>
</kml>
"""
        kml_path = os.path.join(records_dir, "flight_path.kml")
        with open(kml_path, 'w') as f:
            f.write(kml_content)
        print(f"已创建KML轨迹文件: {kml_path}")
    except Exception as e:
        print(f"创建KML文件时出错: {e}")

# ======================================================================
# 主程序执行
# ======================================================================
if __name__ == "__main__":
    X, Y, Z, vegetation, buildings = generate_terrain(size=300, octaves=8, sigma=15)
    construction_area = define_construction_area(X, Y, Z)
    drone_path = generate_drone_path(construction_area, fixed_height_m=1000.0, num_points=150)
    animate_drone_movement(X, Y, Z, vegetation, buildings, construction_area, drone_path)
                