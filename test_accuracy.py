import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ================= 1. 准备“期末考试卷”（测试集） =================
# ⚠️ 注意：这里换成了 test 文件夹！
test_dir = r"D:\MedicalData\chest_xray\test"

# 测试集的预处理必须和训练集一模一样
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
# 测试的时候不需要随机打乱，一车一车正常考就行
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"✅ 考卷准备完毕，共有 {len(test_dataset)} 道题（张图）。")

# ================= 2. 把昨天那个“学生”请出来 =================
# 注意：为了保证代码能直接跑，我把模型结构再贴一遍
import torch.nn as nn


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

# ================= 3. 开始考试！ =================
# 🌟 开启考试模式
model.eval()

correct = 0  # 答对的题数
total = 0  # 总题数

print("\n📝 开始批改试卷...")

# 🌟 告诉 PyTorch：现在是在考试，不要计算梯度（不改错题），省点力气！
with torch.no_grad():
    for images, labels in test_loader:
        # 1. 模型答题（前向传播）
        outputs = model(images)

        # 2. 看看模型选了哪个答案
        # outputs 是 [得分0, 得分1]，我们取得分高的那个作为预测结果
        _, predicted = torch.max(outputs.data, 1)

        # 3. 统计总分
        total += labels.size(0)
        # 如果预测等于真实标签，correct 加 1
        correct += (predicted == labels).sum().item()

# ================= 4. 公布成绩 =================
accuracy = 100 * correct / total
print(f"🎯 考试结束！模型答对了 {correct} 道题，答错了 {total - correct} 道题。")
print(f"🔥 最终准确率 (Accuracy): {accuracy:.2f}%")