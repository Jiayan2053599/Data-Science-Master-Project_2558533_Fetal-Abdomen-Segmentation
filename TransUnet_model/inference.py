"""
Created on 2025/8/6 22:00
@author: 18310
"""

import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import matplotlib.patches as mpatches

# === 配置参数 ===
IS_BINARY_MODE       = True   # True = “伪二分类”可视化 & 打印 All-FG；False = 纯三分类
ENABLE_VISUALIZATION = True   # 是否在屏幕上显示 Top5 叠加图

# === 路径配置 ===
val_img_dir  = r"D:\TransUnet\usdata\val\image"
val_mask_dir = r"D:\TransUnet\usdata\val\mask"
model_path   = r"D:\TransUnet\trainingrecords_transUnet\checkpoint\mydata_transunet_Lovasz\mydata_transunet_Lovasz_0.5189_21.pkl"

base_save_dir    = r"D:\TransUnet\inference_results-Transunet"
mode_folder      = 'binary' if IS_BINARY_MODE else 'multi'
save_pred_dir    = os.path.join(base_save_dir, mode_folder, 'pred_masks')
save_overlay_dir = os.path.join(base_save_dir, mode_folder, 'overlay_top5')
save_csv_path    = os.path.join(base_save_dir, mode_folder, 'metrics.csv')

os.makedirs(save_pred_dir,    exist_ok=True)
os.makedirs(save_overlay_dir, exist_ok=True)

# === 导入模型 ===
sys.path.append(r"D:\TransUnet\ussegcodes")
from models.transunet import TransUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransUNet(
    img_dim=512,
    in_channels=1,
    out_channels=64,
    head_num=4,
    mlp_dim=512,
    block_num=8,
    patch_dim=16,
    class_num=3
).to(device)

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# === 预处理 ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

# === 指标函数 ===
def dice_score(pred, target, smooth=1e-5):
    pred   = (pred   > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum()
    return (2. * inter + smooth) / (union + smooth)

def iou_score(pred, target, smooth=1e-5):
    pred   = (pred   > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter  = (pred & target).sum()
    union  = (pred | target).sum()
    return (inter + smooth) / (union + smooth)

# === 后处理 & 叠加函数 ===
def postprocess_mask(mask, smooth_kernel_size=5, area_thresh=100):
    kernel       = np.ones((smooth_kernel_size, smooth_kernel_size), np.uint8)
    mask_closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_filled  = binary_fill_holes(mask_closed > 0).astype(np.uint8) * 255
    if mask_filled.ndim > 2:
        mask_filled = mask_filled[...,0]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    clean_mask = np.zeros_like(mask_filled, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
            clean_mask[labels==i] = 255
    return clean_mask

def create_overlay(image, pred_mask, return_rgb=True):
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    color_mask = np.zeros_like(image_bgr, dtype=np.uint8)
    # 二分类时 pred_mask 只有 0/255；三分类时仍可显示红/绿
    color_mask[pred_mask == 100] = [0, 0, 255]
    color_mask[pred_mask == 255] = [0, 255, 0]
    overlay = cv2.addWeighted(image_bgr, 0.7, color_mask, 0.3, 0)
    if return_rgb:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

# === 推理主循环 ===
results = []
file_names = sorted(os.listdir(val_img_dir))

for fname in tqdm(file_names, desc="Inferencing"):
    img = cv2.imread(os.path.join(val_img_dir, fname), cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(os.path.join(val_mask_dir, fname), cv2.IMREAD_GRAYSCALE)

    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)

    # —— 统一用 softmax 拿到三类概率 ——
    probs   = torch.softmax(out, dim=1)[0].cpu().numpy()  # (3, H, W)
    fg_prob = probs[1] + probs[2]                         # 合并类1+类2

    # —— 二分类逻辑（All-FG） vs 纯三分类逻辑 ——
    if IS_BINARY_MODE:
        # “伪二分类”：只取前景
        pred_mask = (fg_prob > 0.5).astype(np.uint8) * 255
        dice      = dice_score(pred_mask, (msk>0).astype(np.uint8)*255)
        iou       = iou_score(pred_mask, (msk>0).astype(np.uint8)*255)
        cls1 = cls2 = None
    else:
        # 纯三分类：还原到 0/100/255 掩码
        pred = np.argmax(probs, axis=0).astype(np.uint8)
        pred_mask = np.zeros_like(pred, dtype=np.uint8)
        pred_mask[pred==1] = 100
        pred_mask[pred==2] = 255
        # 后处理整体前景
        merged = postprocess_mask((pred_mask>0).astype(np.uint8)*255)
        clean  = np.zeros_like(pred_mask)
        clean[(merged>0)&(pred_mask==100)] = 100
        clean[(merged>0)&(pred_mask==255)] = 255
        pred_mask = clean

        dice = dice_score(pred_mask, msk)
        iou  = iou_score(pred_mask, msk)
        # 分类别指标
        cls1 = dice_score((pred_mask==100).astype(np.uint8), (msk==100).astype(np.uint8))
        cls2 = dice_score((pred_mask==255).astype(np.uint8), (msk==255).astype(np.uint8))

    results.append({
        "filename":  fname,
        "dice":      dice,
        "iou":       iou,
        "cls1_dice": cls1,
        "cls2_dice": cls2,
        "image":     img,
        "pred_mask":pred_mask
    })
    cv2.imwrite(os.path.join(save_pred_dir, fname), pred_mask)

# === Top-5 可视化 ===
valid = sorted(results, key=lambda x: x["dice"], reverse=True)[:5]

for i, r in enumerate(valid, start=1):
    # 二分类模式下，把任意非零都当前景绿色
    if IS_BINARY_MODE:
        disp_mask = ((r["pred_mask"]>0).astype(np.uint8))*255
    else:
        disp_mask = r["pred_mask"]

    overlay = create_overlay(r["image"], disp_mask, return_rgb=True)
    cv2.imwrite(
        os.path.join(save_overlay_dir, f"{i}_{r['dice']:.4f}_{r['filename']}"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    if ENABLE_VISUALIZATION:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(overlay)
        ax.set_title(f"Top {i} - Dice: {r['dice']:.4f}", fontsize=12)
        ax.axis("off")
        if IS_BINARY_MODE:
            fg = mpatches.Patch(color='green', label='Foreground')
            ax.legend(handles=[fg], loc='lower right')
        else:
            p1 = mpatches.Patch(color='red',   label='Class 1')
            p2 = mpatches.Patch(color='green', label='Class 2')
            ax.legend(handles=[p1,p2], loc='lower right')
        plt.tight_layout()
        plt.show()

# === 保存 CSV & 打印指标 ===
df = pd.DataFrame([{
    "filename":  r["filename"],
    "dice":      r["dice"],
    "iou":       r["iou"]
} for r in results])
df.to_csv(save_csv_path, index=False)

print("\nInference Complete.")
if IS_BINARY_MODE:
    print(f"All-FG Mean Dice: {df['dice'].mean():.4f}")
    print(f"All-FG Mean IoU:   {df['iou'].mean():.4f}")
else:
    print("【Multi-Class】")
    print(f"All-FG Mean Dice: {df['dice'].mean():.4f}")
    print(f"All-FG Mean IoU:   {df['iou'].mean():.4f}")
    print(f"Class1 Mean Dice: {np.nanmean([r['cls1_dice'] for r in results]):.4f}")
    print(f"Class2 Mean Dice: {np.nanmean([r['cls2_dice'] for r in results]):.4f}")
