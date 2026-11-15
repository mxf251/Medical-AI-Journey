import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 设置训练集路径 (注意路径指向包含 NORMAL 和 PNEUMONIA 的那一级文件夹)
data_dir = r"D:\MedicalData\chest_xray\train"

# 2. 定义数据预处理流水线 (Transforms)
# 这里我们直接使用 PyTorch 官方封装好的工具，取代之前手写的 OpenCV 代码
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # 统一缩放成 224x224
    transforms.Grayscale(num_output_channels=1), # 变成单通道灰度图
    transforms.ToTensor(),               # 自动把图像变成 Tensor，并自动将像素值归一化到 0-1 之间！
])

# 3. 召唤 Dataset (数据集)
# ImageFolder 会自动识别文件夹名：NORMAL 作为 0 类，PNEUMONIA 作为 1 类
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"✅ 成功加载数据集！共有 {len(train_dataset)} 张图片。")
print(f"✅ 类别映射关系: {train_dataset.class_to_idx}") # 打印 0 和 1 分别代表什么

# 4. 召唤 DataLoader (数据加载器)
# batch_size=8 表示一次抓 8 张图片出来；shuffle=True 表示每次抓的顺序打乱
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 5. 抽出第一车（第一批）数据看看长什么样
images, labels = next(iter(train_loader))

print("\n--- 抽检第一批数据 ---")
print(f"这批图片的 Tensor 尺寸: {images.shape}")
# 期待输出: torch.Size([8, 1, 224, 224]) -> 8张图，1个通道，224高，224宽
print(f"这批图片的标签: {labels}")
# 期待输出: tensor([0, 1, 0, 1, 1, 1, 0, 0]) 类似这样的 0和1