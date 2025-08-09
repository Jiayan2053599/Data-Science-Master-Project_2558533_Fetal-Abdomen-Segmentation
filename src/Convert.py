"""
Created on 2025/7/2
Updated: 逐帧处理，低内存版本。训练集含正样本，测试集含正+负样本。
"""

import os
import json
import random
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# ---------------- 路径配置 ---------------- #
IMAGES_DIR = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\images\stacked-fetal-ultrasound")
MASKS_DIR = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
SAVE_DIR = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal2D")

imagesTr = SAVE_DIR / "imagesTr"
labelsTr = SAVE_DIR / "labelsTr"
imagesTs = SAVE_DIR / "imagesTs"
labelsTs = SAVE_DIR / "labelsTs"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)
labelsTs.mkdir(parents=True, exist_ok=True)

# ---------------- 参数 ---------------- #
MAX_NEG_PER_CASE = 5
NEG_SAMPLE_RATIO = 0.2  # 测试集中负样本采样比例（例如 20%）

# ---------------- Step 1: 病例划分 ---------------- #
all_case_files = sorted(IMAGES_DIR.glob("*.mha"))
random.seed(42)
random.shuffle(all_case_files)

num_test = int(0.2 * len(all_case_files))
test_files = all_case_files[:num_test]   # 从所有病例文件中抽取前 num_test 个作为测试集（确保划分是按病例进行的，且不会与训练集重叠）
train_files = all_case_files[num_test:]  # 将剩余的病例文件用作训练集

# ---------------- 计数器 ---------------- #
train_list = []
test_list = []
train_total_slices = 0
test_negative_slices = 0
test_positive_slices = 0  # 记录测试集中正样本帧数


# ---------------- Step 2: 处理训练集（仅正样本） ---------------- #
print("处理训练集...")
for img_file in tqdm(train_files):
    case_id = img_file.stem
    msk_file = MASKS_DIR / f"{case_id}.mha"
    if not msk_file.exists():
        continue

    img_itk = sitk.ReadImage(str(img_file))
    msk_itk = sitk.ReadImage(str(msk_file))

    img_arr = sitk.GetArrayFromImage(img_itk)  # [z,h,w]
    msk_arr = sitk.GetArrayFromImage(msk_itk)

    pos_idx = [i for i in range(msk_arr.shape[0]) if np.count_nonzero(msk_arr[i]) > 0]
    if not pos_idx:
        continue

    img_pos = img_arr[pos_idx]
    msk_pos = msk_arr[pos_idx]

    sitk.WriteImage(sitk.GetImageFromArray(img_pos.astype(np.float32)), str(imagesTr / f"{case_id}_0000.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(msk_pos.astype(np.uint8)), str(labelsTr / f"{case_id}.nii.gz"))

    train_list.append({
        "image": f"./imagesTr/{case_id}_0000.nii.gz",
        "label": f"./labelsTr/{case_id}.nii.gz"
    })
    train_total_slices += len(pos_idx)

# ---------------- Step 3: 处理测试集（正样本 + 负样本） ---------------- #
print("处理测试集...")
for img_file in tqdm(test_files):
    case_id = img_file.stem
    msk_file = MASKS_DIR / f"{case_id}.mha"
    if not msk_file.exists():
        continue

    img_itk = sitk.ReadImage(str(img_file))
    msk_itk = sitk.ReadImage(str(msk_file))

    img_arr = sitk.GetArrayFromImage(img_itk)
    msk_arr = sitk.GetArrayFromImage(msk_itk)

    pos_idx = [i for i in range(msk_arr.shape[0]) if np.count_nonzero(msk_arr[i]) > 0]
    neg_idx = [i for i in range(msk_arr.shape[0]) if np.count_nonzero(msk_arr[i]) == 0]

    test_positive_slices += len(pos_idx)
    if not pos_idx:
        continue

    # 采样一部分负样本
    neg_sample_count = min(
        int(len(neg_idx) * NEG_SAMPLE_RATIO),
        len(neg_idx),
        MAX_NEG_PER_CASE
    )
    neg_sampled = random.sample(neg_idx, neg_sample_count)
    test_negative_slices += len(neg_sampled)

    selected_idx = pos_idx + neg_sampled
    selected_idx.sort()

    img_sel = img_arr[selected_idx]
    msk_sel = msk_arr[selected_idx]

    sitk.WriteImage(sitk.GetImageFromArray(img_sel.astype(np.float32)), str(imagesTs / f"{case_id}_0000.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(msk_sel.astype(np.uint8)), str(labelsTs / f"{case_id}.nii.gz"))

    test_list.append(f"./imagesTs/{case_id}_0000.nii.gz")

# ---------------- Step 4: 写入 dataset.json ---------------- #
dataset_json = {
    "name": "Dataset300_ACOptimalSuboptimal2D",
    "description": "Train: positive only; Test: positive + sampled negative",
    "tensorImageSize": "2D",
    "modality": {"0": "ULTRASOUND"},
    "channel_names": {"0": "ultrasound"},
    "labels": {
        "background": 0,
        "optimal_surface": 1,
        "suboptimal_surface": 2
    },
    "file_ending": ".nii.gz",
    "numTraining": len(train_list),
    "numTest": len(test_list),
    "training": train_list,
    "test": test_list
}

with open(SAVE_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

# ---------------- 汇总信息 ---------------- #
# ---------------- 汇总信息 ---------------- #
print("数据预处理完成")
print(f"训练病例数: {len(train_list)}")
print(f"测试病例数: {len(test_list)}")
print(f"训练正样本总帧数: {train_total_slices}")
print(f"测试正样本帧数: {test_positive_slices}")
print(f"测试负样本帧数: {test_negative_slices}")
print(f"dataset.json 已写入: {SAVE_DIR / 'dataset.json'}")

