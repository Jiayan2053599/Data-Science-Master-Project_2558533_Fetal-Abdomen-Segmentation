"""

Created on 2025/7/17 20:23
@author: 18310

"""

from pathlib import Path
import numpy as np

# 您的预处理后影像目录
INPUT_PATH = Path("../test/input/filtered/images/stacked-fetal-ultrasound")
MASK_PATH  = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")

# 枚举所有帧路径，并打标签：1=有掩码，0=无掩码
img_paths = sorted(INPUT_PATH.glob("*.mha")) + sorted(INPUT_PATH.glob("*.tiff"))
labels = []
for p in img_paths:
    # 如果同名掩码存在，则该帧为正样本
    labels.append(1 if (MASK_PATH / p.name).exists() else 0)


# Dataset & Data Loader
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from torchvision import transforms

class FrameDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = sitk.ReadImage(str(self.paths[idx]))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)  # shape (Z,H,W) or (H,W)
        if arr.ndim == 3:  # 若是3D，取中间帧或先平截
            arr = arr[arr.shape[0]//2]
        # 归一化到 [0,1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        # 转成 C×H×W
        arr = np.expand_dims(arr, 0)
        tensor = torch.from_numpy(arr)
        if self.tf:
            tensor = self.tf(tensor)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

# 简单的数据增强／变换
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

# 划分训练/验证
split = int(len(img_paths) * 0.8)
train_ds = FrameDataset(img_paths[:split], labels[:split], transform=train_tf)
val_ds   = FrameDataset(img_paths[split:], labels[split:], transform=None)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

