"""

Created on 2025/7/30 22:11
@author: 18310

"""

"""
终端sh运行命令：
export nnUNet_raw="/home/ec2-user/nnUNet/nnUNet_raw_data_base"
export nnUNet_results="/home/ec2-user/nnUNet/results"
export nnUNet_preprocessed="/home/ec2-user/nnUNet/nnUNet_preprocessed"

nohup python /home/ec2-user/ACOUSLIC-AI-baseline/src/Subset_select_1000/2d_inference_visualize.py > infer.log 2>&1 &
tail -f infer.log
"""
import os
import shutil
from pathlib import Path
import subprocess
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import cv2


# ===== 限制线程数 =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["nnUNet_num_processes"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["nnUNet_num_processes_preprocessing"] = "1"
os.environ["nnUNet_num_processes_segmentation_export"] = "1"


# ===== 参数控制 - 2分类或3分类 =====
# BINARY_MODE = True    # True: 二分类（前景 vs 背景）
BINARY_MODE = False # False: 三分类（类别1 vs 2）


# ===== 数据路径设置 =====
DATASET_ID = "Dataset300_ACOptimalSuboptimal2D"
CONFIG = "2d"
FOLD = "0"
TRAINER = "MyTrainer"

RAW_BASE = Path(os.environ.get("nnUNet_raw", "/home/ec2-user/nnUNet/nnUNet_raw_data_base"))
RESULT_BASE = Path(os.environ.get("nnUNet_results", "/home/ec2-user/nnUNet/results"))

INFER_INPUT_DIR = RAW_BASE / DATASET_ID / "imagesTs"
GT_LABEL_DIR = RAW_BASE / DATASET_ID / "labelsTs"
PRED_OUTPUT_DIR = RESULT_BASE / DATASET_ID / "infer_output" if BINARY_MODE else RESULT_BASE / DATASET_ID / "infer_output_multiclass"
VIS_DIR = RESULT_BASE / DATASET_ID / "infer_vis_binary" if BINARY_MODE else RESULT_BASE / DATASET_ID / "infer_vis_multiclass"

PRED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ===== 推理 =====
print("执行 nnUNetv2_predict ...")
subprocess.run([
    "nnUNetv2_predict",
    "-i", str(INFER_INPUT_DIR),
    "-o", str(PRED_OUTPUT_DIR),
    "-d", DATASET_ID,
    "-c", CONFIG,
    "-f", FOLD,
    "-tr", TRAINER,
    "--disable_tta",
    "--verbose"
], check=True)

# ===== 可视化颜色映射 =====
if BINARY_MODE:
    cmap = mcolors.ListedColormap(['black', 'lime'])  # 背景, 前景
    bounds = [0, 0.5, 1.5]
else:
    cmap = mcolors.ListedColormap(['black', 'lime', 'red'])  # 背景, 类别1, 类别2
    bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# ===== 后处理函数 =====
def post_process_mask_safe(pred_slice, min_area=30, apply_closing=True, smooth=True):
    assert pred_slice.ndim == 2, f"预测输入应为2D数组，但获得了维度 {pred_slice.shape}"
    if BINARY_MODE:
        class_val = int(np.max(pred_slice))
        if class_val == 0:
            return pred_slice
        pred_uint8 = (pred_slice == class_val).astype(np.uint8)
    else:
        pred_uint8 = (pred_slice > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_uint8, connectivity=8)
    if num_labels <= 1:
        return pred_slice

    cleaned = np.zeros_like(pred_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if smooth:
        blurred = cv2.GaussianBlur(closed.astype(np.float32), (9, 9), sigmaX=2)
        smoothed = (blurred > 0.3).astype(np.uint8)
    else:
        smoothed = closed

    if BINARY_MODE:
        return smoothed * class_val
    else:
        return pred_slice

# ===== 评估指标函数 =====
def dice_coefficient(pred, gt, smooth=1e-6):
    inter = np.sum((pred==1)&(gt==1))
    p_sum = np.sum(pred==1)
    g_sum = np.sum(gt==1)
    dice  = (2*inter + smooth) / (p_sum + g_sum + smooth)
    # print(f"DEBUG → I={inter}, P={p_sum}, G={g_sum}, dice={dice:.4f}")
    return dice

def iou_score(pred, gt, smooth=1e-6):
    intersection = np.sum((pred == 1) & (gt == 1))
    union = np.sum((pred == 1) | (gt == 1))
    return (intersection + smooth) / (union + smooth)

# ===== 可视化叠加图像 =====
def visualize_overlay(image_path: Path, pred_path: Path, save_path: Path):
    img_np = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    pred_np = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

    # === 修复：确保图像是2D ===
    if img_np.ndim == 3:
        img_np = img_np[0]
    if pred_np.ndim == 3:
        pred_np = pred_np[0]

    if np.count_nonzero(pred_np) == 0:
        print(f"[跳过] 无预测区域: {image_path.name}")
        return None

    pred_slice = post_process_mask_safe(pred_np)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np, cmap='gray')
    plt.imshow(pred_slice, cmap=cmap, norm=norm, alpha=0.4)
    plt.title(image_path.name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return pred_slice

# ===== 批量处理与评估 =====
print("开始生成叠加图像与评估指标...")
dice_list = []
iou_list = []
dice_class = {1: [], 2: []}
iou_class = {1: [], 2: []}

for pred_file in tqdm(sorted(PRED_OUTPUT_DIR.glob("*.nii.gz"))):
    name       = pred_file.stem.replace(".nii", "")
    image_file = INFER_INPUT_DIR / f"{name}_0000.nii.gz"
    label_file = GT_LABEL_DIR  / f"{name}.nii.gz"
    vis_file   = VIS_DIR       / f"{name}_overlay.png"

    if not image_file.exists():
        print(f"[跳过] 无原图: {image_file}")
        continue
    if not label_file.exists():
        print(f"[跳过] 缺失标签: {label_file}")
        continue

    # —— 先读预测 nii，并生成纯 0/1 mask
    pred_vol  = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file)))
    pred_mask = (pred_vol > 0).astype(np.uint8)

    # —— 读真实标签，同样做成 0/1 mask
    gt_vol  = sitk.GetArrayFromImage(sitk.ReadImage(str(label_file)))
    gt_mask = (gt_vol > 0).astype(np.uint8)

    # —— 单独做可视化，不影响评估
    _ = visualize_overlay(image_file, pred_file, vis_file)

    # —— 真正的评估部分
    if BINARY_MODE:
        dice = dice_coefficient(pred_mask, gt_mask)
        iou = iou_score(pred_mask, gt_mask)
        dice_list.append(dice)
        iou_list.append(iou)
        print(f"{name} – Binary Dice: {dice:.4f}, IoU: {iou:.4f}")

    else:
        for class_id in [1, 2]:
            pred_c = (pred_vol == class_id).astype(np.uint8)
            gt_c = (gt_vol == class_id).astype(np.uint8)
            if pred_c.sum() == 0 and gt_c.sum() == 0:
                continue
            dice = dice_coefficient(pred_c, gt_c)
            iou = iou_score(pred_c, gt_c)
            dice_class[class_id].append(dice)
            iou_class[class_id].append(iou)
            print(f"{name} – 类别 {class_id} Dice: {dice:.4f}, IoU: {iou:.4f}")

# ===== 输出结果 =====
print("\n====== 评估结果 ======")
if BINARY_MODE:
    if dice_list:
        print(f"平均 Binary Dice: {np.mean(dice_list):.4f}")
        print(f"平均 Binary IoU: {np.mean(iou_list):.4f}")
    else:
        print("无可用标签用于评估")
else:
    for cid in [1, 2]:
        if dice_class[cid]:
            print(f"类别 {cid} 平均 Dice: {np.mean(dice_class[cid]):.4f}")
            print(f"类别 {cid} 平均 IoU: {np.mean(iou_class[cid]):.4f}")
        else:
            print(f"类别 {cid} 无可用样本")

print(f"\n推理与可视化完成！输出目录：{PRED_OUTPUT_DIR} | 可视化图：{VIS_DIR}")

# ===== 保存评估结果 =====
metrics_file = RESULT_BASE / DATASET_ID / ("metrics_binary.txt" if BINARY_MODE else "metrics_multiclass.txt")

with open(metrics_file, "w") as f:
    f.write("===== Evaluation Summary =====\n")

    if BINARY_MODE:
        if dice_list:
            mean_dice = np.mean(dice_list)
            mean_iou = np.mean(iou_list)
            f.write(f"Mean Dice: {mean_dice:.4f}\n")
            f.write(f"Mean IoU : {mean_iou:.4f}\n")
            for name, dice, iou in zip(sorted([p.stem for p in PRED_OUTPUT_DIR.glob("*.nii.gz")]), dice_list, iou_list):
                f.write(f"{name}: Dice={dice:.4f}, IoU={iou:.4f}\n")
        else:
            f.write("No valid samples for evaluation.\n")

    else:
        for cid in [1, 2]:
            if dice_class[cid]:
                mean_dice = np.mean(dice_class[cid])
                mean_iou = np.mean(iou_class[cid])
                f.write(f"Class {cid} Mean Dice: {mean_dice:.4f}\n")
                f.write(f"Class {cid} Mean IoU : {mean_iou:.4f}\n")
            else:
                f.write(f"Class {cid}: No valid samples.\n")

print(f"\n评估指标写入成功：{metrics_file}")
