import torch
import torch.nn as nn


# 1. 定义我们的网络结构 (继承自 nn.Module 这个官方模板)
class SimpleMedicalCNN(nn.Module):
    def __init__(self):
        super(SimpleMedicalCNN, self).__init__()

        # 第一层：卷积层 (提取边缘、轮廓等特征)
        # 输入1个通道(灰度图)，输出16个通道(16种不同的特征滤镜)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # 激活函数：把负数变成0，保留正数（过滤无用信息）
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层：把图片长宽砍掉一半 (224 -> 112)

        # 第二层：再加一层卷积
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 再次砍半 (112 -> 56)

        # 全连接层 (做最后的决定)
        # 经过前面两层，图片尺寸变成了 32通道 x 56高 x 56宽
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 最后输出2个数字，代表“正常”和“肺炎”的概率

    # 2. 数据怎么往前走 (前向传播)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # 过第一道工序
        x = self.pool2(self.relu2(self.conv2(x)))  # 过第二道工序

        # 把立体的数据拍扁，变成一条线 (为了塞进全连接层)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ================= 测试环节 =================
if __name__ == "__main__":
    # 1. 实例化我们的模型
    model = SimpleMedicalCNN()
    print("✅ 模型搭建成功！结构如下：")
    print(model)

    # 2. 模拟你上一节课传出来的那批数据 (8张图，1通道，224x224)
    fake_batch_images = torch.rand(8, 1, 224, 224)

    # 3. 把数据喂给模型（这就是未来模型“看”图的过程）
    output = model(fake_batch_images)

    print("\n--- 测试结果 ---")
    print(f"输入模型的图片尺寸: {fake_batch_images.shape}")
    print(f"模型输出的结果尺寸: {output.shape}")
    print("输出内容的数字 (代表2个类别的得分):\n", output)