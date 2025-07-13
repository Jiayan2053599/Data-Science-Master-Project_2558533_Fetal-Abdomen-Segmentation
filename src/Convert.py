"""

Created on 2025/7/2 15:18
@author: 18310

"""
"""
convert.py

一次性完成：
  1) .mha → .nii.gz 并组织 raw_data_base
  2) 生成 nnU-Net v2 要求的 dataset.json
  3) （可选）注册给 nnU-Net v2
  4) 完整执行 fingerprint+planning+preprocessing
"""

import os
import subprocess
import json
from pathlib import Path
import SimpleITK as sitk

# ─── 用户配置 ────────────────────────────────────────────────────────
IMAGES_DIR        = Path(r"E:\Data Science Master Project-code\ACOUSLIC-AI-baseline\test\input\images\stacked-fetal-ultrasound")
MASKS_DIR         = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
RAW_BASE          = Path(r"D:\nnUNet\nnUNet_raw_data_base")
PREPROC_BASE      = Path(r"D:\nnUNet\nnUNet_preprocessed")
RESULTS_BASE      = Path(r"D:\nnUNet\results")

TASK_ID           = 300
TASK_FOLDER_NAME  = "Dataset300_ACOptimalSuboptimal"
PLANS_NAME        = "nnUNetPlans"
# ────────────────────────────────────────────────────────────────────────

def convert_mha_to_nii(src: Path, dst: Path):
    """用 SimpleITK 把 .mha 读入并写出为 .nii.gz（开启压缩）。"""
    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst), useCompression=True)

def prepare_raw_data():
    """
    步骤1: 在 RAW_BASE 下创建
        RAW_BASE / TASK_FOLDER_NAME /
            imagesTr/
            labelsTr/
        并把 .mha -> .nii.gz, 生成 dataset.json
    """
    task_folder = RAW_BASE / TASK_FOLDER_NAME
    imagesTr    = task_folder / "imagesTr"
    labelsTr    = task_folder / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    case_ids = []
    # 转影像
    for mha in sorted(IMAGES_DIR.glob("*.mha")):
        case = mha.stem
        dst = imagesTr / f"{case}_0000.nii.gz"
        if dst.exists():
            print(f"[skip-img] {dst.name} 已存在")
        else:
            print(f"[img  ] 转换 {mha.name} → {dst.name}")
            convert_mha_to_nii(mha, dst)
        case_ids.append(case)

    # 转标签
    for case in case_ids:
        src = MASKS_DIR / f"{case}.mha"
        dst = labelsTr / f"{case}.nii.gz"
        if not src.exists():
            print(f"[warn ] 未找到 {src.name}，跳过")
            continue
        if dst.exists():
            print(f"[skip-msk] {dst.name} 已存在")
        else:
            print(f"[msk  ] 转换 {src.name} → {dst.name}")
            convert_mha_to_nii(src, dst)

    # 写 dataset.json
    dataset = {
        "name": TASK_FOLDER_NAME,
        "description": "Fetal abdomen segmentation",
        "tensorImageSize": "3D",

        # 以下 4 项 nnU-Net v2 强依赖
        "modality": {"0": "BMODE"},
        "channel_names": {"0": "BMODE"},
        "file_ending": ".nii.gz",
        # regions_class_order 列出所有 label 编号（包括 background 0）
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
            }
            for c in case_ids
        ],
        "test": []
    }
    json_path = RAW_BASE / TASK_FOLDER_NAME / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"[json ] 写入 dataset.json → {json_path}")

def register_old_dataset():
    task_folder = RAW_BASE / TASK_FOLDER_NAME
    # 只要我们自己已经把 .mha→.nii.gz、dataset.json 全都准备好了，就不需要再跑转换
    if task_folder.exists():
        print("[skip] Raw data folder 已存在，跳过老版数据集转换")
        return
    # 如果真有特别需求要跑转换，请在这里解开下面的注释
    """
    cmd = [
        "nnUNetv2_convert_old_nnUNet_dataset",
        str(task_folder),
        TASK_FOLDER_NAME
    ]
    print(">> Converting old dataset for nnU-Net v2:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(">> Conversion done.")
    """

def run_planning_preprocessing():
    """
    步骤2: 调用 nnUNetv2_plan_and_preprocess 完成 fingerprint→planning→preprocessing
    输出在 PREPROC_BASE / TASK_FOLDER 下：
      - dataset_fingerprint.json
      - plans.pkl
      - imagesTr/*.npz, labelsTr/*.npz
    """
    pp_task    = PREPROC_BASE / TASK_FOLDER_NAME
    fingerprint = pp_task / "dataset_fingerprint.json"
    imgs_npz    = list((pp_task / "imagesTr").glob("*.npz")) \
                  if (pp_task / "imagesTr").exists() else []

    # 如果 fingerprint + 至少一个 npz 存在，就跳过
    if fingerprint.exists() and imgs_npz:
        print(f"[skip ] 预处理已完成，跳过 → {pp_task}")
        return

    # 确保输出目录存在
    pp_task.mkdir(parents=True, exist_ok=True)

    # 继承父进程的环境变量，并注入 nnUNet 路径
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(RAW_BASE)
    env["nnUNet_preprocessed"] = str(PREPROC_BASE)
    env["nnUNet_results"]      = str(RESULTS_BASE)

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",  str(TASK_ID),                  # 只传整数 ID
        "--verify_dataset_integrity",         # 验证 dataset.json
        "-np",  "4",                          # fingerprint 并行数
        "-npfp","4",                          # preprocessing 并行数
        "--verbose",                          # 显示详细日志
    ]

    print(">> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print("[done] Planning & preprocessing 完成")

if __name__ == "__main__":
    # 先保证根目录存在
    for d in (RAW_BASE, PREPROC_BASE, RESULTS_BASE):
        d.mkdir(parents=True, exist_ok=True)

    # 执行转换 + dataset.json
    prepare_raw_data()
    # 执行 registration + fingerprint+planning+preprocessing
    run_planning_preprocessing()
