"""

Created on 2025/7/2 21:14
@author: 18310

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dataset.py

手动验证 nnU-Net raw_data_base/TaskXXX_… 下的数据集是否符合要求：
 1) 目录结构
 2) dataset.json 中必须字段
 3) 每例训练影像与标签文件一一对应
 4) 影像/标签的 shape、spacing、origin、direction 完全一致
 5) 无 NaN，标签值只在 {0,1}（或你的 labels 中定义的那些整型）里
"""

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ─── 修改为你自己的 Task 文件夹 ─────────────────────────────────
TASK_FOLDER = Path(r"D:\nnUNet\nnUNet_raw_data_base\Task300_Dataset300_ACOptimalSuboptimal")
# ─────────────────────────────────────────────────────────────────

def error(msg):
    print("ERROR:", msg)
    sys.exit(1)

def warn(msg):
    print("WARNING:", msg)

def main():
    # 0) 基本存在性
    if not TASK_FOLDER.exists():
        error(f"找不到任务文件夹: {TASK_FOLDER}")
    ds_file = TASK_FOLDER / "dataset.json"
    if not ds_file.is_file():
        error(f"缺少 dataset.json: {ds_file}")

    # 1) 读 JSON，检查字段
    js = json.loads(ds_file.read_text())
    required = ["labels","channel_names","numTraining","file_ending","training"]
    for k in required:
        if k not in js:
            error(f"dataset.json 缺少必填字段 “{k}”")
    # 允许 modalities/description 等额外存在

    labels_dict    = js["labels"]           # e.g. {"0":"background","1":"abdomen"}
    channel_names  = js["channel_names"]    # e.g.  {"0":"BMODE"}
    num_training   = js["numTraining"]
    file_ending    = js["file_ending"]      # e.g. ".nii.gz"
    training_list  = js["training"]         # list of {"image":..., "label":...}

    # 2) 数量一致性
    if len(training_list) != num_training:
        error(f"numTraining={num_training}，但 'training' 条目数={len(training_list)} 不一致！")

    # 3) 遍历每例
    for rec in training_list:
        img_rel = rec["image"]
        seg_rel = rec["label"]
        img_f = TASK_FOLDER / img_rel
        seg_f = TASK_FOLDER / seg_rel

        # 文件存在性
        if not img_f.is_file():
            error(f"缺少训练影像: {img_f}")
        if not seg_f.is_file():
            error(f"缺少对应标签: {seg_f}")

        # 读取影像与标签
        img = sitk.ReadImage(str(img_f))
        seg = sitk.ReadImage(str(seg_f))

        arr = sitk.GetArrayFromImage(img)   # shape: (Z, H, W)
        lbl = sitk.GetArrayFromImage(seg)

        # 4) NaN 检查
        if np.any(np.isnan(arr)):
            error(f"影像包含 NaN: {img_f}")
        if np.any(np.isnan(lbl)):
            error(f"标签包含 NaN: {seg_f}")

        # 5) Shape/spacing/origin/direction 必须一致
        if arr.shape != lbl.shape:
            error(f"Shape 不匹配: {img_f} vs {seg_f} → {arr.shape} vs {lbl.shape}")
        if not np.allclose(img.GetSpacing(), seg.GetSpacing()):
            error(f"Spacing 不匹配: {img_f} vs {seg_f} → {img.GetSpacing()} vs {seg.GetSpacing()}")
        if not np.allclose(img.GetOrigin(), seg.GetOrigin()):
            warn(f"Origin 不匹配: {img_f} vs {seg_f} → {img.GetOrigin()} vs {seg.GetOrigin()}")
        if not np.allclose(img.GetDirection(), seg.GetDirection()):
            warn(f"Direction 不匹配: {img_f} vs {seg_f} → {img.GetDirection()} vs {seg.GetDirection()}")

        # 6) 标签值检查
        uniques = np.unique(lbl)
        allowed = set(int(k) for k in labels_dict.keys())
        bad = [int(x) for x in uniques if int(x) not in allowed]
        if bad:
            error(f"标签值包含意外类别 {bad}，允许的：{sorted(allowed)} — 文件 {seg_f}")

    print("\n验证通过！你的数据集目录和文件格式都满足 nnU-Net 要求。")

if __name__ == "__main__":
    main()
