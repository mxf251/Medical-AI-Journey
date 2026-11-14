import os
import cv2
import matplotlib.pyplot as plt

# 🌟【关键修改点】把下面引号里的路径，换成你刚才复制的那串文件夹路径！
# 注意：Windows路径前面一定要加一个小写字母 r，并且用双斜杠或者单斜杠
image_dir = r"D:\MedicalData\chest_xray\train\NORMAL"

# 1. 获取这个文件夹里的所有图片名字
image_names = os.listdir(image_dir)

# 2. 我们只拿第一张图片来测试
first_image_name = image_names[0]
# 拼接出这张图片的完整路径
image_path = os.path.join(image_dir, first_image_name)
print(f"正在用代码读取: {first_image_name}")

# 3. 用 OpenCV 读取这张图
img = cv2.imread(image_path)

# 4. 如果读取成功，就把它显示出来
if img is None:
    print("报错：路径不对，没找到图片！请检查路径。")
else:
    print(f"读取成功！这张图的数字矩阵尺寸是: {img.shape}")

    # 5. 颜色转换（因为 OpenCV 默认是 BGR，显示出来会发蓝，转成 RGB 才正常）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 6. 显示图片
    plt.imshow(img_rgb)
    plt.title("Real Chest X-Ray")
    plt.axis('off')
    plt.show()