import os
import sys
import numpy as np
from pathlib import Path
import torch
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class MaskManager:
    def __init__(self, mask_path):
        self.mask_path = mask_path
        self.mask = None
        self.mask_loaded = False
    
    def load_mask(self):
        if self.mask_path and os.path.exists(self.mask_path):
            mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                return False
            
            # 处理不同格式的掩码图像
            if len(mask.shape) == 3:
                if mask.shape[2] == 4:
                    _, mask = cv2.threshold(mask[:, :, 3], 1, 255, cv2.THRESH_BINARY)
                else:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            else:
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            self.mask = mask
            self.mask_loaded = True
            return True
        return False
    
    def resize_mask(self, img_shape):
        if self.mask is None or not self.mask_loaded:
            return None
        
        if (self.mask.shape[0], self.mask.shape[1]) != (img_shape[0], img_shape[1]):
            resized_mask = cv2.resize(self.mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
            return resized_mask
        return self.mask

def is_outside_mask(xyxy, mask):
    if mask is None:
        return True
    
    x_center = int((xyxy[0] + xyxy[2]) / 2)
    y_center = int((xyxy[1] + xyxy[3]) / 2)
    
    if 0 <= y_center < mask.shape[0] and 0 <= x_center < mask.shape[1]:
        return mask[y_center, x_center] == 0
    return True

def process_image(source_path, mask_path, weights_path=None, redline_mode=True):
    """
    处理图像并返回结果图片数组
    
    参数:
        source_path (str): 输入图片路径
        mask_path (str): 掩码图片路径
        weights_path (str): 模型权重路径
        redline_mode (bool): 是否启用红线模式
    
    返回:
        numpy array: 处理后的图片数组
        int: 检测到的违规目标数量
    """
    # 设置默认权重路径
    if weights_path is None:
        weights_path = "/home/gdut-627/4T_hard_disk/gyj/drone_yolov8/ultralytics-8.3.135/runs/detect/train11/weights/best.pt"
    
    # 加载YOLOv8模型（强制使用CPU）
    model = YOLO(weights_path)
    
    # 初始化掩码管理器
    mask_manager = MaskManager(mask_path)
    if redline_mode:
        mask_manager.load_mask()

    # 读取图像
    im0 = cv2.imread(source_path)
    if im0 is None:
        return None, 0
    
    # 获取当前图像匹配的掩码
    current_mask = None
    if redline_mode and mask_manager.mask_loaded:
        current_mask = mask_manager.resize_mask(im0.shape)
        
        # 使用红色轮廓线替代半透明覆盖
        if current_mask is not None:
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im0, contours, -1, (0, 0, 255), 5)  # 红色轮廓
            cv2.drawContours(im0, contours, -1, (0, 0, 0), 2)    # 黑色内轮廓
    
    # YOLOv8推理（强制使用CPU）
    results = model.predict(
        im0,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        max_det=1000,
        device='cpu',
        verbose=False
    )
    
    # 处理检测结果
    violation_count = 0
    line_thickness = 3
    annotator = Annotator(im0, line_width=line_thickness, example=str(model.names))
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        
        for xyxy, conf, cls in zip(boxes, confs, clss):
            label = model.names[cls]
            
            # 施工红线模式：只处理红线外的目标
            if redline_mode:
                if not is_outside_mask(xyxy, current_mask):
                    continue
                
                color = (0, 255, 255)  # 黄色：违规目标
                violation_count += 1
            else:
                color = colors(cls, True)
            
            # 添加标注
            annotator.box_label(xyxy, f"{label} {conf:.2f}", color=color)
    
    return im0
####################################测试用例############
def main():
    # 写死的路径配置
    source_path = "/home/gdut-627/mxxxxxx/0018.jpg"
    mask_path = "/home/gdut-627/mxxxxxx/0018_mask.png"
    weights_path = "/home/gdut-627/4T_hard_disk/gyj/drone_yolov8/ultralytics-8.3.135/runs/detect/train11/weights/best.pt"
    
    # 调用处理函数
    result_img = process_image(
        source_path=source_path,
        mask_path=mask_path,
        weights_path=weights_path,
        redline_mode=True
    )
    
    # 在run函数之外添加保存功能
    if result_img is not None:
        # 保存结果图像
        save_path = "/home/gdut-627/mxxxxxx/detection_result.jpg"
        cv2.imwrite(save_path, result_img)
        
        # 添加换行符\n
        print("\n处理完成！结果已保存至:", save_path)
        
    else:
        print("\n处理失败！请检查输入文件路径\n")

if __name__ == "__main__":
    main()