import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
import pickle
from pathlib import Path
from random import shuffle

# === 配置路径 ===
DATASET_NAME = "Dataset300_ACOptimalSuboptimal2D"
RAW_BASE = Path("D:/nnUNet/nnUNet_raw_data_base")
PREPROCESSED_BASE = Path("D:/nnUNet/nnUNet_preprocessed")
imagesTr = RAW_BASE / DATASET_NAME / "imagesTr"
labelsTr = RAW_BASE / DATASET_NAME / "labelsTr"
imagesTs = RAW_BASE / DATASET_NAME / "imagesTs"
labelsTs = RAW_BASE / DATASET_NAME / "labelsTs"
dataset_json_path = RAW_BASE / DATASET_NAME / "dataset.json"
splits_path = PREPROCESSED_BASE / DATASET_NAME / "splits_final.pkl"
MAX_NUM = 10  # 从验证集抽取多少帧用于推理测试

# === 创建测试文件夹
imagesTs.mkdir(parents=True, exist_ok=True)
labelsTs.mkdir(parents=True, exist_ok=True)

# === 加载 splits_final.pkl 获取验证集 ID
with open(splits_path, "rb") as f:
    splits = pickle.load(f)
val_ids = splits[0]["val"]  # 使用第一个fold

print(f"验证集包含 {len(val_ids)} 个病例")

# === 搜集验证集中的所有图像切片路径
candidate_slices = []
for cid in val_ids:
    img_path = imagesTr / f"{cid}_0000.nii.gz"
    label_path = labelsTr / f"{cid}.nii.gz"
    if not (img_path.exists() and label_path.exists()):
        continue

    # 加载标签图像并筛选出有标注的帧索引
    label_img = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    for i in range(label_img.shape[0]):
        if np.count_nonzero(label_img[i]) > 0:
            candidate_slices.append((cid, i))

print(f"验证集中共找到 {len(candidate_slices)} 张有标注的帧")

# === 打乱并抽取推理帧
shuffle(candidate_slices)
selected_slices = candidate_slices[:MAX_NUM]
print(f"选中 {len(selected_slices)} 张帧用于推理测试")

selected_ids = []

for cid, slice_idx in selected_slices:
    # 从原始图像和标签中读取该帧
    img_path = imagesTr / f"{cid}_0000.nii.gz"
    label_path = labelsTr / f"{cid}.nii.gz"
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
    lbl_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))

    frame_img = sitk.GetImageFromArray(img_arr[slice_idx][np.newaxis, :, :])
    frame_lbl = sitk.GetImageFromArray(lbl_arr[slice_idx][np.newaxis, :, :])

    new_name = f"{cid}_{slice_idx:04d}"
    sitk.WriteImage(frame_img, str(imagesTs / f"{new_name}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(frame_lbl, str(labelsTs / f"{new_name}.nii.gz"), useCompression=True)
    selected_ids.append(new_name)
    print(f"保存推理帧: {new_name}")

# === 更新 dataset.json 中 test_cases 字段
with open(dataset_json_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

dataset["test_cases"] = selected_ids

with open(dataset_json_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"推理测试集构建完成，更新了 dataset.json 中 test_cases 字段")
