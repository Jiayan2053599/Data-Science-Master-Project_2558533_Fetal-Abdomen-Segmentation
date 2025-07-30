"""

Created on 2025/7/28 04:13
@author: 18310

"""
import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# ---------------- 用户设置 ----------------
IMG_DIR = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\images\stacked-fetal-ultrasound")
LBL_DIR = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
OUT_DIR = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset050_AC_mha2d_top2000")
MAX_SAMPLES = 2000
# ----------------------------------------

# 自动创建输出目录
imagesTr = OUT_DIR / "imagesTr"
labelsTr = OUT_DIR / "labelsTr"
imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

print(f"开始提取前 {MAX_SAMPLES} 个含标签的 2D 切片...\n")

sample_count = 0
training_entries = []

for img_fn in tqdm(sorted(os.listdir(IMG_DIR)), desc="扫描 .mha volumes"):
    if not img_fn.endswith(".mha"):
        continue

    base_id = img_fn[:-4]
    img_path = IMG_DIR / img_fn
    lbl_path = LBL_DIR / f"{base_id}.mha"

    if not lbl_path.exists():
        continue

    img3d = sitk.ReadImage(str(img_path))
    lbl3d = sitk.ReadImage(str(lbl_path))
    D = img3d.GetSize()[2]

    for i in range(D):
        if sample_count >= MAX_SAMPLES:
            break

        img_slice = img3d[:, :, i]
        lbl_slice = lbl3d[:, :, i]
        lbl_array = sitk.GetArrayFromImage(lbl_slice)

        if np.any(lbl_array > 0):  # 有标签
            sid = f"{base_id}_{i:04d}"
            img_out = imagesTr / f"{sid}_0000.nii.gz"
            lbl_out = labelsTr / f"{sid}.nii.gz"
            sitk.WriteImage(img_slice, str(img_out), useCompression=True)
            sitk.WriteImage(lbl_slice, str(lbl_out), useCompression=True)

            training_entries.append({
                "image": f"./imagesTr/{sid}_0000.nii.gz",
                "label": f"./labelsTr/{sid}.nii.gz"
            })
            sample_count += 1

    if sample_count >= MAX_SAMPLES:
        break

print(f" 成功提取 {sample_count} 张含标签的 2D 切片")

# ---------------- 写入 dataset.json ----------------
def stringify_keys(d):
    return {str(k): v for k, v in d.items()}

dataset_json = {
    "name": "Dataset050_AC_mha2d_positive_top2000",
    "description": "Top 2000 2D slices with positive label from .mha volumes",
    "tensorImageSize": "2D",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "modality": {
        "0": "US"
    },
    "channel_names": {
        "0": "US"
    },
    "labels": {
        "background": 0,
        "abdomen_best": 1,
        "abdomen_suboptimal": 2
    },
    "file_ending": ".nii.gz",
    "numTraining": len(training_entries),
    "numTest": 0,
    "training": training_entries,
    "test": []
}

with open(OUT_DIR / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=2, ensure_ascii=False)

print(f" dataset.json 写入完成，共 {len(training_entries)} 条训练记录。")
print(f" 路径: {OUT_DIR / 'dataset.json'}")
