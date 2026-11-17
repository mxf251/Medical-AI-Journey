import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random  # 🌟 第一处：新增这一行

# ================= 1. 准备食材（数据和模型） =================
# 数据路径 (换成你自己的)
data_dir = r"D:\MedicalData\chest_xray\train"

# 预处理流水线
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
# 为了方便快速看到效果，我们只取前 64 张图来跑（真实训练是取全部）
#small_subset = torch.utils.data.Subset(train_dataset, range(64))
# 🌟 第二处：改成下面这三行随机抽取
random_indices = random.sample(range(len(train_dataset)), 64)
small_subset = torch.utils.data.Subset(train_dataset, random_indices)
train_loader = DataLoader(small_subset, batch_size=8, shuffle=True)


# ================= 2. 复制昨天搭好的大脑（模型） =================
class SimpleMedicalCNN(nn.Module):
    def __init__(self):
        super(SimpleMedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = SimpleMedicalCNN()

# ================= 3. 设定奖惩机制 =================
# 损失函数（算错题）：交叉熵，专门用来衡量分类错误有多离谱
criterion = nn.CrossEntropyLoss()

# 优化器（老师）：Adam算法，负责根据错误来修改大脑里的参数，学习率设为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= 4. 开始上课（训练循环） =================
print("🚀 开始训练！看看 Loss 怎么变化：\n")

# 让模型进入“训练模式”
model.train()

epochs = 3  # 把所有数据轮着看 3 遍
for epoch in range(epochs):
    running_loss = 0.0

    # 从传送带上拿出一车一车的数据
    for batch_idx, (images, labels) in enumerate(train_loader):

        # 动作1：清空过去的错误记忆（梯度清零）
        optimizer.zero_grad()

        # 动作2：前向传播（大脑看图，给出预测得分）
        outputs = model(images)

        # 动作3：算错题（对比预测得分和真实标签，算出误差）
        loss = criterion(outputs, labels)

        # 动作4：反向传播（根据误差，找出是谁的责任）
        loss.backward()

        # 动作5：优化更新（老师改作业，纠正大脑里的参数）
        optimizer.step()

        # 记录当前这一车的误差值
        running_loss += loss.item()

        # 每跑完 4 车（32张图），打印一下当前的误差
        if (batch_idx + 1) % 4 == 0:
            print(f"第 {epoch + 1} 轮, 第 {batch_idx + 1} 车 | 当前 Loss (误差): {loss.item():.4f}")

print("\n🎉 训练结束！你刚刚完成了 AI 的第一次进化！")