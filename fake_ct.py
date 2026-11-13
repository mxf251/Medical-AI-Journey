import numpy as np
import matplotlib.pyplot as plt

# 1. 创建一个 256 x 256 的矩阵，里面的数字是随机的，模拟图像的像素点
# 乘以 255 是因为灰度图的像素值范围是 0-255
fake_image = np.random.rand(256, 256) * 255

# 2. 使用 matplotlib 画图
# cmap='gray' 表示用灰度模式（黑白）显示，这是医学图像的标配
plt.imshow(fake_image, cmap='gray')

# 3. 加个标题
plt.title("My First Fake Medical Image")

# 4. 隐藏坐标轴（让图看起来更干净）
plt.axis('off')

# 5. 把图显示出来
plt.show()