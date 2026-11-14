import os

import cv2
import matplotlib.pyplot as plt
image_dir = r"D:\MedicalData\chest_xray\train\NORMAL"


image_names = os.listdir(image_dir)


first_image_name = image_names[0]

image_path = os.path.join(image_dir, first_image_name)
print(f"正在用代码读取: {first_image_name}")

img = cv2.imread(image_path)
print(f"原图尺寸: {img.shape}")#这里打印出来应该还是原先的大尺寸
if img is not None:
    #动作1：褪色——将BRG彩色图转为灰度图（单通道）
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #动作2：瘦身——将图片强行缩放成224*224的大小
    resized_img = cv2.resize(gray_img,(224,224))

    #动作3：归一化——把像素值从0-255压缩到0-1之间——更快更稳定
    normalized_img = resized_img / 255.0

    print(f"预处理后的尺寸：: {normalized_img.shape}")
    print(f"像素值的范围：{normalized_img.min()}到{normalized_img.max()}")

#cmap='gray'必须加，否则颜色会变
    plt.imshow(normalized_img, cmap='gray')
    plt.title("Preprocessed X-Ray (224*224)")
    plt.axis('off')
    plt.show()
