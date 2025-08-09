"""

Created on 2025/7/30 20:05
@author: 18310

"""
"""
Created on 2025/7/30 20:05
@author: 18310
"""

import os
import json
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === 路径设置 ===
TRAIN_DS = "Dataset047_ACOptimalSuboptimal_slice2000_2d"
# TRAIN_DS = "Dataset050_AC_mha2d_top2000"
nnunet_raw = Path(r"D:/nnUNet/nnUNet_raw_data_base")
train_ds_dir = nnunet_raw / TRAIN_DS
imagesTr = train_ds_dir / "imagesTr"
imagesTs = train_ds_dir / "imagesTs"
dataset_json = train_ds_dir / "dataset.json"
# plans_path = Path(r"D:/nnUNet/nnUNet_preprocessed/Dataset050_AC_mha2d_top2000/nnUNetPlans.json")
plans_path = Path(r"D:\nnUNet\nnUNet_preprocessed\Dataset047_ACOptimalSuboptimal_slice2000_2d\nnUNetPlans.json")

mha_img_dir = Path(r"D:/Data_Science_project-data/acouslic-ai-train-set/images/stacked-fetal-ultrasound")
mha_seg_dir = Path(r"D:/Data_Science_project-data/acouslic-ai-train-set/masks/stacked_fetal_abdomen")

imagesTs.mkdir(exist_ok=True)

# === 从训练记录中提取已用 ID ===
train_ids = {f.name.split("_")[0] for f in imagesTr.glob("*.gz") if "_" in f.name}

# === 读取计划文件，获取 spacing 和归一化参数 ===
with open(plans_path, 'r') as f:
    plans = json.load(f)

TARGET_SPACING = plans['configurations']['2d']['spacing']
MEAN = 73.3181
STD = 28.0098

# === 查找未见数据 ===
unseen_mha_files = [f for f in mha_img_dir.glob("*.mha") if f.stem not in train_ids]

# 控制最大帧数
MAX_TOTAL_FRAMES = 10
saved_ids = []
total_saved = 0

for mha_path in unseen_mha_files:
    if total_saved >= MAX_TOTAL_FRAMES:
        break

    cid = mha_path.stem
    seg_path = mha_seg_dir / f"{cid}.mha"
    if not seg_path.exists():
        continue

    img3d = sitk.ReadImage(str(mha_path))
    seg3d = sitk.ReadImage(str(seg_path))
    spacing3d = img3d.GetSpacing()
    direction3d = img3d.GetDirection()
    origin3d = img3d.GetOrigin()

    img_np = sitk.GetArrayFromImage(img3d)
    seg_np = sitk.GetArrayFromImage(seg3d)

    for i in range(img_np.shape[0]):
        if total_saved >= MAX_TOTAL_FRAMES:
            break

        if np.any(seg_np[i] > 0):
            raw_slice = sitk.GetImageFromArray(img_np[i])
            raw_slice.SetSpacing(spacing3d[:2])
            raw_slice.SetOrigin(origin3d[:2])

            try:
                raw_slice.SetDirection(direction3d[:4])
            except:
                raw_slice.SetDirection((1.0, 0.0, 0.0, 1.0))

            # === 重采样至 TARGET_SPACING ===
            new_size = [
                int(np.round(raw_slice.GetSize()[j] * raw_slice.GetSpacing()[j] / TARGET_SPACING[j]))
                for j in range(2)
            ]
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(TARGET_SPACING)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(raw_slice.GetDirection())
            resampler.SetOutputOrigin(raw_slice.GetOrigin())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled = resampler.Execute(raw_slice)

            # === Z-score 标准化 ===
            arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
            arr = (arr - MEAN) / STD
            norm_img = sitk.GetImageFromArray(arr)
            norm_img.CopyInformation(resampled)

            sid = f"{cid}_{i:04d}"
            out_path = imagesTs / f"{sid}_0000.nii.gz"
            sitk.WriteImage(norm_img, str(out_path))

            saved_ids.append(sid)
            total_saved += 1
            print(f"[{total_saved}] 保存推理图像: {sid}_0000.nii.gz")

# === 更新 dataset.json ===
with open(dataset_json, "r", encoding="utf-8") as f:
    ds = json.load(f)

ds.pop("test", None)
ds["test_cases"] = saved_ids

with open(dataset_json, "w", encoding="utf-8") as f:
    json.dump(ds, f, indent=2)

print(f"[✔] 共保存 {total_saved} 个测试帧并更新 dataset.json。")


# from pathlib import Path
# import SimpleITK as sitk
#
# imagesTs = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset050_AC_mha2d_top2000\imagesTs")
#
# for f in imagesTs.glob("*.nii.gz"):
#     img = sitk.ReadImage(str(f))
#     arr = sitk.GetArrayFromImage(img)
#     print(f"{f.name} shape: {arr.shape}")



