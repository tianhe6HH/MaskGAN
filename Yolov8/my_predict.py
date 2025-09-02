'''
    用于将YOLO-seg的预测结果转换为细胞掩码图像，而且不是boxes的哦~
'''
import cv2
from sympy.printing.pretty.pretty_symbology import line_width
from ultralytics import YOLO
from PIL import Image
import os
import re
import numpy as np
from PIL import Image, ImageDraw
import warnings
import concurrent.futures
warnings.filterwarnings("ignore", category=FutureWarning)
img_size = (256, 256)

if __name__ == "__main__":
    # 指定包含图像文件的文件夹地址
    directory_path = './data'
    # 检查文件夹是否存在
    if not os.path.exists(directory_path):
        print(f"指定的文件夹 {directory_path} 不存在。")
    else:
        # 加载预训练的 YOLO 模型
        model = YOLO("best.pt")  # pretrained YOLOv8n model
        # 对每个图像文件进行预测
        results = model.predict(directory_path, save=False, show=False, imgsz=256, co15nf=0.,
                                show_labels=False, show_conf=False, show_boxes=False, line_width=None)
        # To predict masks for every image in files
        for result in results:
            if result.masks is not None:
                h, w = result.orig_img.shape[:2]  # 获取原始图像的尺寸
                mask_image = np.zeros((h, w), dtype=np.uint8)
                # 获取所有掩码
                masks = result.masks.data.cpu().numpy()
                # 遍历所有掩码并绘制到 mask_image 上
                for mask in masks:
                    resized_mask = cv2.resize(mask, (w, h))
                    resized_mask = (resized_mask * 255).astype(np.uint8)
                    # 将掩码合并到 mask_image 上
                    mask_image = cv2.bitwise_or(mask_image, resized_mask)
                cv2.imwrite(os.path.join(directory_path,'result.png'), mask_image)

