"""
Created on 2025/7/13 14:59
@author: 18310
"""

import json
import os
from glob import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from model import FetalAbdomenSegmentation, select_fetal_abdomen_mask_and_frame

# 限制 ITK 只用 1 个线程
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

# 配置路径
INPUT_PATH = Path("../test/input/filtered/images/stacked-fetal-ultrasound")
MASK_PATH  = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
OUTPUT_PATH = Path("../test/selected_output_batch")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 参数：要处理的正样本数量上限
MAX_SAMPLES = 5

# 评估函数

def load_mask(path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(bool)


def dice_coeff(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    vol = pred.sum() + gt.sum()
    return 2 * inter / vol if vol > 0 else 1.0


def iou_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    uni = np.logical_or(pred, gt).sum()
    return inter / uni if uni > 0 else 1.0


def precision_recall(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    p = tp / (tp + fp) if tp + fp > 0 else 1.0
    r = tp / (tp + fn) if tp + fn > 0 else 1.0
    return p, r

def evaluate_metrics(mapping: list) -> list:
    records = []
    for item in mapping:
        # 读取 3D 预测与真值掩码
        pred3d = load_mask(Path(item['segmentation']))
        img_name = Path(item['image']).name
        gt3d = load_mask(MASK_PATH / img_name)

        # 模型选帧索引
        pred_frame_idx = item.get('frame_idx')
        # 确保索引有效
        if pred_frame_idx is None or pred_frame_idx < 0 or pred_frame_idx >= pred3d.shape[0]:
            continue

        # 提取对应帧的 2D 掩码
        pred2d = pred3d[pred_frame_idx]
        gt2d   = gt3d[pred_frame_idx]

        # 计算指标
        d = dice_coeff(pred2d, gt2d)
        j = iou_score(pred2d, gt2d)
        p, r = precision_recall(pred2d, gt2d)

        records.append({
            'sample_id':     item['sample_id'],
            'frame_idx':     pred_frame_idx,
            'dice_2d':       round(d, 4),
            'iou_2d':        round(j, 4),
            'precision_2d':  round(p, 4),
            'recall_2d':     round(r, 4)
        })
    return records


# 实例化算法
algorithm = FetalAbdomenSegmentation()
_ShowTorchInfo = lambda: None

# 筛选正样本
pos_files = []
for p in sorted(INPUT_PATH.glob("*.mha")) + sorted(INPUT_PATH.glob("*.tiff")):
    if (MASK_PATH / p.name).exists():
        pos_files.append(p)
pos_files = pos_files[:MAX_SAMPLES]

# 批量推理
mapping = []
for idx, img_path in enumerate(pos_files, 1):
    print(f"Processing {idx}/{len(pos_files)}: {img_path.name}")
    prob = algorithm.predict(str(img_path), save_probabilities=True)
    post = algorithm.postprocess(prob)
    mask2d, frame_idx = select_fetal_abdomen_mask_and_frame(post)

    seg_dir = OUTPUT_PATH / f"sample_{idx}"
    seg_dir.mkdir(exist_ok=True)

    # 构建 3D 掩码
    depth = prob.shape[1]
    mask3d = np.zeros((depth, *mask2d.shape), dtype=np.uint8)
    if frame_idx >= 0:
        mask3d[frame_idx] = mask2d

    out_mask = seg_dir / 'segmentation.mha'
    sitk.WriteImage(sitk.GetImageFromArray(mask3d), str(out_mask), True)

    # 保存帧索引
    with open(seg_dir / 'frame_index.json', 'w') as f:
        json.dump({'frame_idx': int(frame_idx)}, f)

    mapping.append({
        'sample_id':   idx,
        'image':       str(img_path),
        'segmentation': str(out_mask),
        'frame_idx':   frame_idx
    })

# 保存 Mapping
with open(OUTPUT_PATH / 'mapping_batch.json', 'w') as f:
    json.dump(mapping, f, indent=4)

# 评估并打印
records = evaluate_metrics(mapping)
print("\nEvaluation Metrics on Predicted Frames:")
for r in records:
    print(r)

# 平均 2D 指标
if records:
    mean_d = sum(r['dice_2d'] for r in records) / len(records)
    mean_j = sum(r['iou_2d']  for r in records) / len(records)
    print(f"Mean 2D Dice: {mean_d:.4f}")
    print(f"Mean 2D IoU : {mean_j:.4f}")