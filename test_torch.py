import torch

print("正在启动 PyTorch...")
print(f"当前 PyTorch 版本: {torch.__version__}")

# 模拟一张 224x224 的图片（3个通道），也就是你之前预处理后的大小
fake_xray_tensor = torch.rand(3, 224, 224)
print(f"X光片 Tensor 的尺寸: {fake_xray_tensor.shape}")

print("🚀 大功告成！PyTorch 环境配置完美！")