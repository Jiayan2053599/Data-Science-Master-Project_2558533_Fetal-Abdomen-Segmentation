import os
import json
import shutil
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==== 配置路径 ====
RAW_IMG_DIR = Path(r"D:/Data_Science_project-data/acouslic-ai-train-set/images/stacked-fetal-ultrasound")
RAW_MASK_DIR = Path(r"D:/Data_Science_project-data/acouslic-ai-train-set/masks/stacked_fetal_abdomen")

OUT_DATASET_NAME = "Dataset303_Top50AnnotatedSubset"
OUT_BASE = Path(r"D:/nnUNet/nnUNet_raw_data_base")
OUT_DIR = OUT_BASE / OUT_DATASET_NAME
IMAGES_TR = OUT_DIR / "imagesTr"
LABELS_TR = OUT_DIR / "labelsTr"
IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)

# ==== 读取所有文件并计算标注帧占比 ====
volume_stats = []
print("统计标注帧占比中...")

for mask_file in tqdm(list(RAW_MASK_DIR.glob("*.mha"))):
    case_id = mask_file.stem
    image_file = RAW_IMG_DIR / f"{case_id}.mha"
    if not image_file.exists():
        print(f"图像不存在: {case_id}")
        continue

    try:
        mask = sitk.ReadImage(str(mask_file))
        mask_array = sitk.GetArrayFromImage(mask)  # shape: [D, H, W]
        total_slices = mask_array.shape[0]
        annotated = np.sum((mask_array > 0).any(axis=(1, 2)))  # 有标注的帧数（label > 0）
        ratio = annotated / total_slices
        volume_stats.append((case_id, annotated, total_slices, ratio))
    except Exception as e:
        print(f"读取失败: {case_id}，错误: {e}")

# ==== 选取前50个标注帧比例最高的 ====
volume_stats.sort(key=lambda x: x[3], reverse=True)
top50 = volume_stats[:50]

print(f"\n 已选取前50个标注帧占比最高的文件。")

# ==== 转换 .mha → .nii.gz 并强制标签为 int64，同时重采样到较低分辨率 ====
print("\n正在转换 .mha → .nii.gz，同时执行重采样...")

def resample(image, new_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)

for case_id, _, _, _ in tqdm(top50):
    in_img = RAW_IMG_DIR / f"{case_id}.mha"
    in_mask = RAW_MASK_DIR / f"{case_id}.mha"
    out_img = IMAGES_TR / f"{case_id}_0000.nii.gz"
    out_mask = LABELS_TR / f"{case_id}.nii.gz"

    try:
        # 读取
        img = sitk.ReadImage(str(in_img))
        mask = sitk.ReadImage(str(in_mask))

        # 重采样（推荐 spacing 可调）
        target_spacing = (1.0, 1.0, 1.0)  # 你可以改成 0.8 或 1.2 以平衡质量与内存
        img_resampled = resample(img, new_spacing=target_spacing, is_label=False)
        mask_resampled = resample(mask, new_spacing=target_spacing, is_label=True)

        # 强制标签为 int64（避免 float 导致报错）
        mask_arr = sitk.GetArrayFromImage(mask_resampled).astype(np.int64)
        mask_img_final = sitk.GetImageFromArray(mask_arr)
        mask_img_final.CopyInformation(mask_resampled)

        # 保存
        sitk.WriteImage(img_resampled, str(out_img))
        sitk.WriteImage(mask_img_final, str(out_mask))

    except Exception as e:
        print(f"转换失败: {case_id}，错误: {e}")

# ==== 写入 dataset.json（添加 label 2）====
print("\ 正在生成 dataset.json...")

dataset_dict = {
    "name": OUT_DATASET_NAME,
    "description": "Top 50 annotated volumes with best and suboptimal planes",
    "tensorImageSize": "3D",
    "modality": {"0": "US"},
    "channel_names": {"0": "US"},
    "file_ending": ".nii.gz",
    "labels": {
        "background": 0,
        "abdomen_best": 1,
        "abdomen_suboptimal": 2
    },
    "regions_class_order": [0, 1, 2],
    "numTraining": len(top50),
    "numTest": 0,
    "training": [
        {
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        }
        for case_id, _, _, _ in top50
    ],
    "test": []
}

with open(OUT_DIR / "dataset.json", "w") as f:
    json.dump(dataset_dict, f, indent=4)

print(f"\n 全部完成！输出路径：{OUT_DIR}")
