"""

Created on 2025/8/2 20:15
@author: 18310

"""

import os
import pickle
from pathlib import Path
from sklearn.model_selection import KFold

# 读取 imagesTr 下的所有图像文件，提取病人ID
imagesTr_dir = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal2D\imagesTr")
case_ids = sorted({f.stem.split('_')[0] for f in imagesTr_dir.glob("*.nii.gz")})

# 5折交叉验证划分
kf = KFold(n_splits=5, shuffle=True, random_state=12345)
splits = []

for train_idx, val_idx in kf.split(case_ids):
    train_cases = [case_ids[i] for i in train_idx]
    val_cases = [case_ids[i] for i in val_idx]
    splits.append({
        'train': train_cases,
        'val': val_cases
    })

# 保存到 nnUNet_preprocessed 对应数据集目录下
splits_path = Path(r"D:\nnUNet\nnUNet_preprocessed\Dataset300_ACOptimalSuboptimal2D\splits_final.pkl")
splits_path.parent.mkdir(parents=True, exist_ok=True)

with open(splits_path, "wb") as f:
    pickle.dump(splits, f)

print(f"已保存 splits_final.pkl，包含 {len(splits)} 个交叉验证折")