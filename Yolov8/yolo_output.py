'''
    将yolo pretrained模型的框图txt结果做成256*256的mask图像
'''
import re
import os
import numpy as np
from PIL import Image, ImageDraw


def list_files_in_directory(directory):
    # 创建一个空列表，用于存储文件名
    files_list = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 将文件名添加到列表中
            files_list.append(file)

    return files_list

# 替换目标目录
directory_path = '../dataset/PamHeRegister20240122/HE_num/train'
files = list_files_in_directory(directory_path)

# 打印所有文件名
for file in files:
    # print(file)
    # 读取文件内容
    with open(r"F:\PythonProgram\UGATIT_PyTorch_Implementation-main\UGATIT_PyTorch_Implementation-main\Yolov8\output.txt", 'r') as file:
        data = file.read()

    # 使用正则表达式提取数值
    pattern = re.compile(r'\[\s*([\d\.\s,\-]+)\s*\]')
    matches = pattern.findall(data)

    # 将匹配结果解析为numpy数组
    boxes = []
    for match in matches:
        boxes.append([float(num) for num in match.split(',')])
    boxes = np.array(boxes)

    # 创建256x256的黑色图像
    image_size = (256, 256)
    image = Image.new('RGB', image_size, 'black')
    draw = ImageDraw.Draw(image)

    # 在图像上绘制标记框
    for box in boxes:
        cx, cy, w, h = box
        left = cx - w / 2
        top = cy - h / 2
        right = cx + w / 2
        bottom = cy + h / 2
        draw.rectangle([left, top, right, bottom], outline='white', fill='white')

    # 保存或显示图像
    image.show()
    # 如果需要保存图像，可以使用以下代码
    image.save(r'F:\PythonProgram\UGATIT_PyTorch_Implementation-main\UGATIT_PyTorch_Implementation-main\Yolov8\output1.png')
    print('image saved in F:_PythonProgram_UGATIT_PyTorch_Implementation-main_UGATIT_PyTorch_Implementation-main_Yolov8_output1.png')
