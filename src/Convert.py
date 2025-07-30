"""

Created on 2025/7/2 15:18
@author: 18310

"""
"""
convert_2d.py

完成：
  1) .mha → .nii.gz 并组织 raw_data_base
  2) 生成 nnU-Net v2 要求的 dataset.json（保持 tensorImageSize 为 3D）
  3) 执行 fingerprint+planning+preprocessing，重点用于 2D 配置
"""

import os
import subprocess
import json
from pathlib import Path
import SimpleITK as sitk

# ─── 用户配置 ────────────────────────────────────────────────────────
IMAGES_DIR        = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\images\stacked-fetal-ultrasound")
MASKS_DIR         = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
RAW_BASE          = Path(r"D:\nnUNet\nnUNet_raw_data_base")
PREPROC_BASE      = Path(r"D:\nnUNet\nnUNet_preprocessed")
RESULTS_BASE      = Path(r"D:\nnUNet\results")

TASK_ID           = 300
TASK_FOLDER_NAME  = "Dataset300_ACOptimalSuboptimal"
# ────────────────────────────────────────────────────────────────────────

def convert_mha_to_nii(src: Path, dst: Path):
    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst), useCompression=True)

def prepare_raw_data():
    task_folder = RAW_BASE / TASK_FOLDER_NAME
    imagesTr    = task_folder / "imagesTr"
    labelsTr    = task_folder / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    case_ids = []
    for mha in sorted(IMAGES_DIR.glob("*.mha")):
    #for mha in sorted(IMAGES_DIR.glob("*.mha"))[:150]:
        case = mha.stem
        dst = imagesTr / f"{case}_0000.nii.gz"
        if not dst.exists():
            print(f"[img  ] {mha.name} → {dst.name}")
            convert_mha_to_nii(mha, dst)
        else:
            print(f"[skip-img] {dst.name} 已存在")
        case_ids.append(case)

    for case in case_ids:
        src = MASKS_DIR / f"{case}.mha"
        dst = labelsTr / f"{case}.nii.gz"
        if src.exists() and not dst.exists():
            print(f"[msk  ] {src.name} → {dst.name}")
            convert_mha_to_nii(src, dst)
        elif not src.exists():
            print(f"[warn ] 未找到 {src.name}，跳过")
        else:
            print(f"[skip-msk] {dst.name} 已存在")

    dataset = {
        "name": TASK_FOLDER_NAME,
        "description": "Fetal abdomen segmentation",
        "tensorImageSize": "3D",     # 关键：不要改成 2D！
        "modality": {"0": "BMODE"},
        "channel_names": {"0": "BMODE"},
        "file_ending": ".nii.gz",
        "regions_class_order": [0, 1, 2],
        "labels": {
            "background": 0,
            "optimal": 1,
            "suboptimal": 2
        },
        "numTraining": len(case_ids),
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/{c}_0000.nii.gz",
                "label": f"./labelsTr/{c}.nii.gz"
            } for c in case_ids
        ],
        "test": []
    }

    with open(task_folder / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"[json ] 写入 dataset.json → {task_folder/'dataset.json'}")

if __name__ == "__main__":
    print("程序启动")
    for d in (RAW_BASE, PREPROC_BASE, RESULTS_BASE):
        d.mkdir(parents=True, exist_ok=True)

    prepare_raw_data()