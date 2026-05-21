"""模拟 web 应用推理路径（不依赖 Django ORM）"""
import os, sys

# 模拟路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification', 'cgfnet_models'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

# 导入模型
from main_models import ResNet50 as CGFNetResNet50
from classification.lwganet_model import LWGANet_L0_1242_e32_k11_GELU as LWGANet_L0

print("=" * 60)
print("Testing CGFNet + UCM (matching web app exactly)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/ours_resnet50_UCM08_model_best.pth.tar'
num_classes = 21

# 创建模型 (和 model_loader.py 完全一致的参数)
model = CGFNetResNet50(
    num_classes=num_classes,
    att_type='ours',
    reduction_ratio=16,
    beta=4.0,
    method_version='original',
    channel_groups=64,
    sa_kernel_size=3,
    no_spatial=False,
    no_lofct=False
)

# 加载权重 (和 model_loader.py 完全一致的逻辑)
checkpoint = torch.load(model_path, map_location=device)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('module.', '').replace('_orig_mod.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)
model.eval()

print(f"Model loaded. best_val={checkpoint.get('best_val_ac_score', '?')}, epoch={checkpoint.get('epoch', '?')}")

# 测试1: 随机噪声
print("\n--- Test 1: Random noise ---")
x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    out = model(x)
probs = F.softmax(out, dim=1)[0]
print(f"Top1: {probs.max().item()*100:.1f}%, class: {probs.argmax().item()}")
print(f"Top5: {[f'{probs.topk(5).values[i].item()*100:.1f}%' for i in range(5)]}")

# 测试2: 模拟真实遥感图像 (用纯色+纹理)
print("\n--- Test 2: Simulated remote sensing patterns ---")
patterns = {
    'uniform_green': torch.ones(3, 300, 300) * torch.tensor([0.3, 0.6, 0.2]).view(3, 1, 1),
    'uniform_gray': torch.ones(3, 300, 300) * 0.5,
    'noise': torch.randn(3, 300, 300),
    'checkerboard': torch.zeros(3, 300, 300),
}

patterns['checkerboard'][:, ::20, ::20] = 1.0

for name, pattern in patterns.items():
    img = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8) * 255
    img = img.to(torch.uint8).numpy().transpose(1, 2, 0)
    pil_img = Image.fromarray(img).convert('RGB')

    # 使用和 model_loader.py 完全一致的 transform
    tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
    probs = F.softmax(out, dim=1)[0]
    top_val, top_idx = probs.max(0)
    print(f"  {name}: top1={top_val.item()*100:.1f}%, class_idx={top_idx.item()}")

# 测试3: 使用训练时完全一致的 transform (Resize((256,256), BILINEAR))
print("\n--- Test 3: Training-identical transform (Resize((256,256), BILINEAR)) ---")
for name, pattern in [('uniform_green', patterns['uniform_green'])]:
    img = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8) * 255
    img = img.to(torch.uint8).numpy().transpose(1, 2, 0)
    pil_img = Image.fromarray(img).convert('RGB')

    tensor = transforms.Compose([
        transforms.Resize((256, 256), Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
    probs = F.softmax(out, dim=1)[0]
    top_val, top_idx = probs.max(0)
    print(f"  uniform_green (training transform): top1={top_val.item()*100:.1f}%, class_idx={top_idx.item()}")

# 测试4: 直接用 CGFNet 训练时的 solver 方式
print("\n--- Test 4: Using CGFNet solver's exact transform class ---")
val_transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for name, pattern in patterns.items():
    img = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8) * 255
    img = img.to(torch.uint8).numpy().transpose(1, 2, 0)
    pil_img = Image.fromarray(img).convert('RGB')

    tensor = val_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    probs = F.softmax(out, dim=1)[0]
    top_val, top_idx = probs.max(0)
    print(f"  {name}: top1={top_val.item()*100:.1f}%, class_idx={top_idx.item()}")
