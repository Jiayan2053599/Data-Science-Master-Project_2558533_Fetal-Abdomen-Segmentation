#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
UNet inference & evaluation aligned with nnUNet settings,
with triptych visualization (Image | Pred Overlay (Best/Sub-best) | Confusion Overlay with legend)
and plane-consistency (majority/minority) analysis & optional enforcement.
"""

import os, sys, cv2, torch, numpy as np, pandas as pd, heapq
from tqdm import tqdm
from torchvision import transforms
from scipy.ndimage import binary_fill_holes

# --- matplotlib (headless safe) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===================== Switches & Params =====================
IS_BINARY_MODE        = False   # False=多分类视图；True=二值All-FG
BINARY_POSTPROCESS    = False   # Binary 分支默认不做形态学（与 nnUNet 对齐）
AGGREGATE_BY_CASE     = True    # 主口径：per-case All-FG
ENABLE_VISUALIZATION  = True    # 保存 overlay & triptych
TOPK_OVERLAYS         = 50      # 叠加图 Top-K
TOPK_TRIPTYCH         = 10      # 三联图 Top-K（全局）
SAVE_PER_CASE_BEST    = True    # 每病例保存最佳一帧三联图（独立于Top-K）
FG_THRESH             = 0.5     # Binary 合并前景阈值
TRIP_ALPHA            = 0.5     # 三联图半透明层透明度

# ---- Plane consistency (only for multi-class) ----
PLANE_SANITY              = True      # 是否统计/执行平面一致化
PLANE_SANITY_SCOPE        = "lcc"     # "all_fg" 或 "lcc"（仅最大连通域上统计/修复）
PLANE_SANITY_ENFORCE      = True      # True=把前景强制改成多数类
PLANE_ENFORCE_MINORITY_GT = 0.00      # minority_share > 该阈值才强制（0 表示只要有混合就修复）
ANNOTATE_PLANE_INFO       = True      # 中间面板角标显示 Majority & mix

# I/O 路径（按需修改）
val_img_dir   = r"D:\TransUnet\usdata\val\image"
val_mask_dir  = r"D:\TransUnet\usdata\val\mask"
model_path    = r"D:\TransUnet\trainingrecords\checkpoint\mydata_unet_Lovasz\mydata_unet_Lovasz_0.7052_48.pkl"
base_save_dir = r"D:\TransUnet\inference_results-UNet"

# 模型源码根（包含 models/unet.py）
sys.path.append(r"D:\TransUnet\ussegcodes")

# 输出目录
mode_folder      = 'binary' if IS_BINARY_MODE else 'multi'
save_pred_dir    = os.path.join(base_save_dir, mode_folder, 'pred_masks')
save_overlay_dir = os.path.join(base_save_dir, mode_folder, 'overlays_topk')
save_trip_dir    = os.path.join(base_save_dir, mode_folder, 'vis_target_comparison')
save_best_dir    = os.path.join(base_save_dir, mode_folder, 'per_case_best')
save_csv_path    = os.path.join(base_save_dir, mode_folder, 'metrics_per_slice.csv')
os.makedirs(save_pred_dir,    exist_ok=True)
os.makedirs(save_overlay_dir, exist_ok=True)
os.makedirs(save_trip_dir,    exist_ok=True)
os.makedirs(save_best_dir,    exist_ok=True)

# ===================== Model =====================
from models.unet import UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=3, in_channels=1).to(device)

def _strip_module(sd):
    if len(sd) and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

try:
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
except TypeError:
    ckpt = torch.load(model_path, map_location=device)

loaded = False
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(_strip_module(ckpt["model_state_dict"]), strict=True); loaded = True
elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    model.load_state_dict(_strip_module(ckpt), strict=True); loaded = True
elif isinstance(ckpt, dict):
    for v in ckpt.values():
        if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
            try:
                model.load_state_dict(_strip_module(v), strict=True); loaded = True; break
            except: pass
if not loaded:
    raise ValueError("未找到可用的 state_dict，请检查权重文件。")
model.eval()

# ===================== Metrics =====================
def dice_score(pred, target, smooth=1e-6):
    pred   = (pred   > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter  = (pred & target).sum()
    return (2. * inter + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    pred   = (pred   > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    inter  = (pred & target).sum()
    union  = (pred | target).sum()
    return (inter + smooth) / (union + smooth)

# ===================== Morphology (for Multi) =====================
def postprocess_mask(bin_mask, smooth_kernel_size=5, area_thresh=100):
    """
    bin_mask: {0,255}, 前景=255. 闭运算 -> 填洞 -> 小连通域剔除
    """
    kernel      = np.ones((smooth_kernel_size, smooth_kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    mask_filled = binary_fill_holes(mask_closed > 0).astype(np.uint8) * 255
    if mask_filled.ndim > 2: mask_filled = mask_filled[..., 0]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    clean_mask = np.zeros_like(mask_filled, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
            clean_mask[labels == i] = 255
    return clean_mask

# ===================== Plane consistency (multi) =====================
def _largest_cc(mask_bool: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label as cc_label
        lab, n = cc_label(mask_bool.astype(np.uint8))
        if n <= 1:
            return mask_bool
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        return (lab == np.argmax(sizes))
    except Exception:
        return mask_bool

def apply_plane_sanity(pred_map, scope="all_fg", enforce=True, enforce_thr=0.0):
    """
    对多类别预测 pred_map(0/1/2) 做平面一致化统计/修复。
    返回:
      pred_fixed, majority_plane, minority_share
        - majority_plane: 0(无前景)/1/2
        - minority_share: min(cnt1,cnt2)/(cnt1+cnt2)；无前景时=0.0
    """
    fg = (pred_map > 0)
    if not np.any(fg):
        return pred_map, 0, 0.0

    roi = fg if scope == "all_fg" else _largest_cc(fg)
    c1 = int(np.sum((pred_map == 1) & roi))
    c2 = int(np.sum((pred_map == 2) & roi))
    total = c1 + c2
    if total == 0:
        return pred_map, 0, 0.0

    majority = 1 if c1 >= c2 else 2
    minority_share = float(min(c1, c2) / total)

    if enforce and (minority_share > enforce_thr):
        out = pred_map.copy()
        out[roi] = majority
        return out, majority, minority_share
    else:
        return pred_map, majority, minority_share

# ===================== Simple overlay (for TopK) =====================
def create_overlay(image, label012):
    """灰度 + {0,1,2}（1=绿, 2=金黄），与三联图风格一致"""
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()
    color = np.zeros_like(base, dtype=np.uint8)
    color[label012 == 1] = [0, 255, 0]       # BGR 绿
    color[label012 == 2] = [0, 215, 255]     # BGR 金黄(255,215,0)
    return cv2.addWeighted(base, 0.7, color, 0.3, 0)

# ===================== Triptych helpers =====================
# 颜色（RGB, 0-1）
COL_BIN = {
    "tp": (0.0, 1.0, 0.0),      # 绿
    "fn": (1.0, 0.0, 0.0),      # 红
    "fp": (0.0, 0.2, 1.0),      # 蓝
}
COL_MULTI = {
    "tp1": (0.0, 1.0, 0.0),     # 绿（Best Plane 正确）
    "tp2": (1.0, 0.843, 0.0),   # 金黄（Sub-best 正确，≈255,215,0）
    "fn1": (1.0, 0.0, 0.0),     # 红（Best Miss）
    "fp1": (0.0, 0.2, 1.0),     # 蓝（Best False）
    "fn2": (1.0, 0.55, 0.0),    # 橙（Sub-best Miss）
    "fp2": (0.66, 0.0, 1.0),    # 紫（Sub-best False）
}

def _rgba_layer(mask_bool: np.ndarray, rgb: tuple, alpha: float):
    """mask_bool -> RGBA 图层 (H,W,4), 颜色=rgb(0-1), 透明度=alpha"""
    h, w = mask_bool.shape
    layer = np.zeros((h, w, 4), dtype=np.float32)
    layer[..., 0] = rgb[0] * mask_bool
    layer[..., 1] = rgb[1] * mask_bool
    layer[..., 2] = rgb[2] * mask_bool
    layer[..., 3] = alpha   * mask_bool
    return layer

def _imshow_gray(ax, gray):
    ax.imshow(gray, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.axis('off')

# ---------- Binary (merged FG) ----------
def save_triptych_binary(gray_u8, gt01, pred01, out_png, alpha=TRIP_ALPHA):
    gb = (gt01   > 0).astype(np.uint8)
    pb = (pred01 > 0).astype(np.uint8)

    tp = (gb & pb).astype(bool)
    fn = ((gb == 1) & (pb == 0))
    fp = ((gb == 0) & (pb == 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _imshow_gray(axes[0], gray_u8); axes[0].set_title("Image", fontsize=22, pad=10)

    _imshow_gray(axes[1], gray_u8)
    axes[1].imshow(_rgba_layer(pb.astype(bool), COL_BIN["tp"], alpha), interpolation='nearest')
    axes[1].set_title("Pred Overlay (Merged)", fontsize=22, pad=10)

    _imshow_gray(axes[2], gray_u8)
    axes[2].imshow(_rgba_layer(tp, COL_BIN["tp"], alpha), interpolation='nearest')
    axes[2].imshow(_rgba_layer(fp, COL_BIN["fp"], alpha), interpolation='nearest')
    axes[2].imshow(_rgba_layer(fn, COL_BIN["fn"], alpha), interpolation='nearest')
    axes[2].set_title("Overlay (TP/FN/FP)", fontsize=22, pad=10)

    handles = [
        Patch(color=COL_BIN["tp"], label="TP (Correct FG)"),
        Patch(color=COL_BIN["fn"], label="FN (Missed FG)"),
        Patch(color=COL_BIN["fp"], label="FP (False FG)"),
    ]
    axes[2].legend(handles=handles, loc='lower right', fontsize=12,
                   framealpha=0.85, facecolor='white', edgecolor='black')

    plt.tight_layout(w_pad=3.5)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------- Multi-class (六色互斥 + 角标) ----------
def save_triptych_multiclass(gray_u8, gt012, pred012, out_png, alpha=TRIP_ALPHA, plane_info=None):
    """
    左：Image；中：Pred Overlay（1=绿，2=金黄）；右：六色互斥 TP1/TP2/FN1/FP1/FN2/FP2
    """
    gt1 = (gt012 == 1); gt2 = (gt012 == 2)
    pr1 = (pred012 == 1); pr2 = (pred012 == 2)

    tp1 = gt1 & pr1;  fn1 = gt1 & (~pr1); fp1 = (~gt1) & pr1
    tp2 = gt2 & pr2;  fn2 = gt2 & (~pr2); fp2 = (~gt2) & pr2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Image
    _imshow_gray(axes[0], gray_u8); axes[0].set_title("Image", fontsize=22, pad=10)

    # Pred Overlay: 1=绿, 2=金黄
    _imshow_gray(axes[1], gray_u8)
    axes[1].imshow(_rgba_layer(pr1, COL_MULTI["tp1"], alpha), interpolation='nearest')
    axes[1].imshow(_rgba_layer(pr2, COL_MULTI["tp2"], alpha), interpolation='nearest')
    axes[1].set_title("Pred Overlay (Best/Sub-best)", fontsize=22, pad=10)

    # 角标：多数类与混合比例
    if ANNOTATE_PLANE_INFO and (plane_info is not None) and (plane_info[0] is not None):
        maj, mix = plane_info
        axes[1].text(0.02, 0.98, f"Majority={maj}  mix={mix:.3f}",
                     transform=axes[1].transAxes, va='top', ha='left',
                     fontsize=11, color='white',
                     bbox=dict(facecolor='black', alpha=0.55, pad=3, edgecolor='none'))

    # Confusion Overlay: 先画错，再画对（避免被覆盖）
    _imshow_gray(axes[2], gray_u8)
    for m, key in [(fn1,'fn1'), (fp1,'fp1'), (fn2,'fn2'), (fp2,'fp2'),
                   (tp1,'tp1'), (tp2,'tp2')]:
        axes[2].imshow(_rgba_layer(m, COL_MULTI[key], alpha), interpolation='nearest')
    axes[2].set_title("Overlay (Error Map)", fontsize=22, pad=10)

    handles = [
        Patch(color=COL_MULTI["tp1"], label="TP1 - Best Plane Correct"),
        Patch(color=COL_MULTI["fn1"], label="FN1 - Best Plane Missed"),
        Patch(color=COL_MULTI["fp1"], label="FP1 - Best Plane False"),
        Patch(color=COL_MULTI["tp2"], label="TP2 - Sub-Best Correct"),
        Patch(color=COL_MULTI["fn2"], label="FN2 - Sub-Best Missed"),
        Patch(color=COL_MULTI["fp2"], label="FP2 - Sub-Best False"),
    ]
    axes[2].legend(handles=handles, loc='lower right', fontsize=12,
                   framealpha=0.85, facecolor='white', edgecolor='black')

    plt.tight_layout(w_pad=3.5)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ===================== Helpers =====================
transform = transforms.ToTensor()
def get_case_id(fname: str) -> str:
    stem = os.path.splitext(fname)[0]
    return stem.split('_')[0]  # 例如 "123_45.png" -> "123"

# ===================== Evaluation containers =====================
rows_slicewise = []                     # [filename, dice, iou, majority_plane, minority_share]
case_scores = {}                        # 合并前景 per-case, All-FG
case_scores_c1, case_scores_c2 = {}, {} # Multi per-class per-case
neg_total_slices = 0
neg_fp_slices    = 0

topk_pool = []                          # for overlays: (dice, case, fname, overlay)
trip_heap = []                          # min-heap for triptych: (dice, idx, case, fname, gray, gt_lbl, pred_lbl, plane_info)
heap_idx  = 0

# ★ 每病例最佳帧缓存：case_id -> (best_dice, record_tuple)
# rec_tuple: (dice, case, fname, gray, gt_lbl, pred_lbl, plane_info)
best_trip_by_case = {}

# 平面一致化统计
mix_count = 0
enforce_count = 0
mix_ratios = []

# ===================== Inference loop (per image) =====================
file_names = sorted([f for f in os.listdir(val_img_dir)
                     if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))])

for fname in tqdm(file_names, desc="Inferencing (UNet, aligned)"):

    img_path = os.path.join(val_img_dir, fname)
    msk_path = os.path.join(val_mask_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        print(f"[Skip] {fname}")
        continue

    case_id = get_case_id(fname)
    case_scores.setdefault(case_id, [])
    case_scores_c1.setdefault(case_id, [])
    case_scores_c2.setdefault(case_id, [])

    with torch.no_grad():
        logits = model(transform(img).unsqueeze(0).to(device))
        if logits.ndim == 3: logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()  # (C,H,W)

    if IS_BINARY_MODE:
        # ---- Binary（合并前景阈值；默认不做形态学）----
        fg_prob  = (probs[1] if probs.shape[0] > 1 else 0) + (probs[2] if probs.shape[0] > 2 else 0)
        pred_bin = (fg_prob > FG_THRESH).astype(np.uint8) * 255
        if BINARY_POSTPROCESS:
            pred_bin = postprocess_mask(pred_bin)

        cv2.imwrite(os.path.join(save_pred_dir, fname), pred_bin)
        gt_bin = (msk > 0).astype(np.uint8) * 255

        d_sw = dice_score(pred_bin, gt_bin)
        i_sw = iou_score(pred_bin,  gt_bin)
        rows_slicewise.append([fname, d_sw, i_sw, np.nan, np.nan])

        if gt_bin.sum() > 0:
            case_scores[case_id].append((d_sw, i_sw))
        else:
            neg_total_slices += 1
            if pred_bin.sum() > 0: neg_fp_slices += 1

        if ENABLE_VISUALIZATION:
            disp012 = np.zeros_like(msk, dtype=np.uint8)
            disp012[pred_bin > 0] = 1
            overlay = create_overlay(img, disp012)
            topk_pool.append((float(d_sw), case_id, fname, overlay))

            gt_lbl   = (gt_bin > 0).astype(np.uint8)    # 0/1
            pred_lbl = (pred_bin > 0).astype(np.uint8)  # 0/1
            plane_info = None
            entry = (float(d_sw), heap_idx, case_id, fname,
                     img.copy(), gt_lbl.copy(), pred_lbl.copy(), plane_info)
            if len(trip_heap) < TOPK_TRIPTYCH: heapq.heappush(trip_heap, entry)
            else:
                if entry[0] > trip_heap[0][0]: heapq.heapreplace(trip_heap, entry)
            heap_idx += 1

            # ★ 更新每病例最佳
            rec = (float(d_sw), case_id, fname, img.copy(), gt_lbl.copy(), pred_lbl.copy(), plane_info)
            if (case_id not in best_trip_by_case) or (d_sw > best_trip_by_case[case_id][0]):
                best_trip_by_case[case_id] = (float(d_sw), rec)

    else:
        # ---- Multi：合并前景→形态学清理→还原类别（与 nnUNet 对齐）----
        am_raw = np.argmax(probs, axis=0).astype(np.uint8)  # 0/1/2
        merged = postprocess_mask(((am_raw > 0).astype(np.uint8) * 255), smooth_kernel_size=5, area_thresh=100)
        am = np.zeros_like(am_raw, dtype=np.uint8)
        am[(merged > 0) & (am_raw == 1)] = 1
        am[(merged > 0) & (am_raw == 2)] = 2

        # ★ 平面一致化：统计/（可选）修复
        maj_plane, mix_ratio = np.nan, np.nan
        if PLANE_SANITY:
            am_fixed, maj_plane, mix_ratio = apply_plane_sanity(
                am, scope=PLANE_SANITY_SCOPE,
                enforce=PLANE_SANITY_ENFORCE,
                enforce_thr=PLANE_ENFORCE_MINORITY_GT
            )
            # 统计
            if mix_ratio is not None and mix_ratio > 0:
                mix_count += 1; mix_ratios.append(float(mix_ratio))
            if PLANE_SANITY_ENFORCE and mix_ratio is not None and mix_ratio > PLANE_ENFORCE_MINORITY_GT:
                enforce_count += 1
            am = am_fixed  # 应用修复

        # 保存多类 mask（0,100,255）
        pred_mask = np.zeros_like(am, dtype=np.uint8)
        pred_mask[am == 1] = 100
        pred_mask[am == 2] = 255
        cv2.imwrite(os.path.join(save_pred_dir, fname), pred_mask)

        # Primary：All-FG（合并前景）
        pred_bin = (am > 0).astype(np.uint8) * 255
        gt_bin   = (msk > 0).astype(np.uint8) * 255

        d_sw = dice_score(pred_bin, gt_bin)
        i_sw = iou_score(pred_bin,  gt_bin)
        rows_slicewise.append([
            fname, d_sw, i_sw,
            (maj_plane if not np.isnan(maj_plane) else np.nan),
            (mix_ratio if not np.isnan(mix_ratio) else np.nan)
        ])

        if gt_bin.sum() > 0:
            case_scores[case_id].append((d_sw, i_sw))
        else:
            neg_total_slices += 1
            if pred_bin.sum() > 0: neg_fp_slices += 1

        # per-class per-case（类内）
        pred1 = (am == 1).astype(np.uint8) * 255
        pred2 = (am == 2).astype(np.uint8) * 255
        gt1   = (msk == 100).astype(np.uint8) * 255
        gt2   = (msk == 255).astype(np.uint8) * 255
        if gt1.sum() > 0:
            case_scores_c1[case_id].append((dice_score(pred1, gt1), iou_score(pred1, gt1)))
        if gt2.sum() > 0:
            case_scores_c2[case_id].append((dice_score(pred2, gt2), iou_score(pred2, gt2)))

        if ENABLE_VISUALIZATION:
            overlay = create_overlay(img, am)
            topk_pool.append((float(d_sw), case_id, fname, overlay))

            gt_lbl = np.zeros_like(msk, dtype=np.uint8)
            gt_lbl[msk == 100] = 1
            gt_lbl[msk == 255] = 2
            pred_lbl = am.astype(np.uint8)
            plane_info = (int(maj_plane) if not np.isnan(maj_plane) else None,
                          float(mix_ratio) if not np.isnan(mix_ratio) else None)
            entry = (float(d_sw), heap_idx, case_id, fname,
                     img.copy(), gt_lbl.copy(), pred_lbl.copy(), plane_info)
            if len(trip_heap) < TOPK_TRIPTYCH: heapq.heappush(trip_heap, entry)
            else:
                if entry[0] > trip_heap[0][0]: heapq.heapreplace(trip_heap, entry)
            heap_idx += 1

            # ★ 更新每病例最佳
            rec = (float(d_sw), case_id, fname, img.copy(), gt_lbl.copy(), pred_lbl.copy(), plane_info)
            if (case_id not in best_trip_by_case) or (d_sw > best_trip_by_case[case_id][0]):
                best_trip_by_case[case_id] = (float(d_sw), rec)

# ===================== Save Top-K overlays =====================
if ENABLE_VISUALIZATION and len(topk_pool) > 0:
    topk_pool.sort(key=lambda x: x[0], reverse=True)
    for rank, (d, case_id, fname, overlay) in enumerate(topk_pool[:TOPK_OVERLAYS], start=1):
        outp = os.path.join(save_overlay_dir, f"{rank:02d}_{d:.4f}_{case_id}_{fname}")
        cv2.imwrite(outp, overlay)

# ===================== Save Top-K triptych =====================
if ENABLE_VISUALIZATION and len(trip_heap) > 0:
    trip_sorted = sorted(trip_heap, key=lambda x: (x[0], x[1]), reverse=True)
    for rank, (d, _, case_id, fname, gray, gt_lbl, pred_lbl, plane_info) in enumerate(trip_sorted, start=1):
        out_png = os.path.join(save_trip_dir, f"{rank:02d}_{d:.4f}_{case_id}_{os.path.splitext(fname)[0]}_triptych.png")
        if IS_BINARY_MODE:
            save_triptych_binary(gray, gt_lbl, pred_lbl, out_png, alpha=TRIP_ALPHA)
        else:
            save_triptych_multiclass(gray, gt_lbl, pred_lbl, out_png, alpha=TRIP_ALPHA, plane_info=plane_info)

# ===================== ★ Save Per-Case Best Triptychs =====================
if ENABLE_VISUALIZATION and SAVE_PER_CASE_BEST and len(best_trip_by_case) > 0:
    rows = []
    for case_id, (d, rec) in sorted(best_trip_by_case.items()):
        _, _, fname, gray, gt_lbl, pred_lbl, plane_info = rec
        out_png = os.path.join(save_best_dir, f"{case_id}_best_{d:.4f}_{os.path.splitext(fname)[0]}_triptych.png")
        if IS_BINARY_MODE:
            save_triptych_binary(gray, gt_lbl, pred_lbl, out_png, alpha=TRIP_ALPHA)
        else:
            save_triptych_multiclass(gray, gt_lbl, pred_lbl, out_png, alpha=TRIP_ALPHA, plane_info=plane_info)
        rows.append([case_id, fname, d])
    pd.DataFrame(rows, columns=["case_id","best_frame","best_dice_allFG"]).to_csv(
        os.path.join(base_save_dir, mode_folder, "per_case_best_list.csv"), index=False
    )

# ===================== Aggregate metrics & summary =====================
df = pd.DataFrame(rows_slicewise,
                  columns=["filename", "dice", "iou", "majority_plane", "minority_share"])
df.to_csv(save_csv_path, index=False)

def _percase_stats(bucket: dict):
    case_d, case_i = [], []
    for _case, lst in bucket.items():
        if not lst: continue
        case_d.append(float(np.mean([d for d, _ in lst])))
        case_i.append(float(np.mean([i for _, i in lst])))
    n = len(case_d)
    mean_d = float(np.mean(case_d)) if n else float("nan")
    mean_i = float(np.mean(case_i)) if n else float("nan")
    sd_d   = float(np.std(case_d, ddof=1)) if n > 1 else float("nan")
    sd_i   = float(np.std(case_i, ddof=1)) if n > 1 else float("nan")
    return mean_d, mean_i, n, sd_d, sd_i

# 主口径：per-case, All-FG
mean_dice_percase_allfg, mean_iou_percase_allfg, n_cases_primary, \
    sd_dice_percase_allfg, sd_iou_percase_allfg = _percase_stats(case_scores)

# 多分类时的 per-class 按病例口径
if not IS_BINARY_MODE:
    mean_dice_pc_c1, mean_iou_pc_c1, n_cases_c1, sd_dice_pc_c1, sd_iou_pc_c1 = _percase_stats(case_scores_c1)
    mean_dice_pc_c2, mean_iou_pc_c2, n_cases_c2, sd_dice_pc_c2, sd_iou_pc_c2 = _percase_stats(case_scores_c2)
else:
    mean_dice_pc_c1 = mean_iou_pc_c1 = sd_dice_pc_c1 = sd_iou_pc_c1 = np.nan; n_cases_c1 = 0
    mean_dice_pc_c2 = mean_iou_pc_c2 = sd_dice_pc_c2 = sd_iou_pc_c2 = np.nan; n_cases_c2 = 0

neg_fpr = float(neg_fp_slices / neg_total_slices) if neg_total_slices > 0 else float("nan")

print("\n====== Evaluation (UNet) ======")
print(f"[Primary] (per-case, All-FG) Dice: {mean_dice_percase_allfg:.4f} ± {sd_dice_percase_allfg:.4f}  (N={n_cases_primary})")
print(f"[Primary] (per-case, All-FG) IoU : {mean_iou_percase_allfg:.4f} ± {sd_iou_percase_allfg:.4f}  (N={n_cases_primary})")
if not IS_BINARY_MODE:
    print(f"[Per-class] C1 Dice: {mean_dice_pc_c1:.4f} ± {sd_dice_pc_c1:.4f} | IoU: {mean_iou_pc_c1:.4f} ± {sd_iou_pc_c1:.4f} | cases: {n_cases_c1}")
    print(f"[Per-class] C2 Dice: {mean_dice_pc_c2:.4f} ± {sd_dice_pc_c2:.4f} | IoU: {mean_iou_pc_c2:.4f} ± {sd_iou_pc_c2:.4f} | cases: {n_cases_c2}")
if PLANE_SANITY and not IS_BINARY_MODE:
    avg_mix = (np.mean(mix_ratios) if len(mix_ratios) else 0.0)
    print(f"[Plane-consistency] mixed_slices={mix_count}, enforced={enforce_count}, avg_mix_ratio={avg_mix:.4f}")
# print(f"[All-slices] Neg-FPR: {neg_fpr:.4f} ({neg_fp_slices}/{neg_total_slices})")

print(f"Overlays(top{TOPK_OVERLAYS}) -> {save_overlay_dir}")
print(f"Triptych(top{TOPK_TRIPTYCH}) -> {save_trip_dir}")
print(f"Per-case Best -> {save_best_dir}")
print(f"Per-slice CSV -> {save_csv_path}")

# ===================== Write summary =====================
summary_path = os.path.join(base_save_dir, mode_folder, "metrics_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== UNet Inference Summary =====\n")
    f.write(f"Mode: {'Binary (All-FG, no postproc)' if IS_BINARY_MODE else 'Multi-class (morph on merged FG)'}\n")
    f.write(f"Images dir: {val_img_dir}\n")
    f.write(f"Masks  dir: {val_mask_dir}\n")
    f.write(f"Model ckpt: {model_path}\n\n")
    f.write(f"[Primary] (per-case, All-FG) Dice: {mean_dice_percase_allfg:.6f} ± {sd_dice_percase_allfg:.6f} (N={n_cases_primary})\n")
    f.write(f"[Primary] (per-case, All-FG) IoU : {mean_iou_percase_allfg:.6f} ± {sd_iou_percase_allfg:.6f} (N={n_cases_primary})\n")
    if not IS_BINARY_MODE:
        f.write(f"[Per-class] C1 Dice: {mean_dice_pc_c1:.6f} ± {sd_dice_pc_c1:.6f} | IoU: {mean_iou_pc_c1:.6f} ± {sd_iou_pc_c1:.6f} | cases: {n_cases_c1}\n")
        f.write(f"[Per-class] C2 Dice: {mean_dice_pc_c2:.6f} ± {sd_dice_pc_c2:.6f} | IoU: {mean_iou_pc_c2:.6f} ± {sd_iou_pc_c2:.6f} | cases: {n_cases_c2}\n")
        f.write(f"[Plane-consistency] mixed_slices={mix_count}, enforced={enforce_count}, avg_mix_ratio={(np.mean(mix_ratios) if len(mix_ratios) else 0.0):.6f}\n")
    f.write(f"Per-slice CSV: {save_csv_path}\n")
    f.write(f"Overlays(top{TOPK_OVERLAYS}): {save_overlay_dir}\n")
    f.write(f"Triptych(top{TOPK_TRIPTYCH}): {save_trip_dir}\n")
    f.write(f"Per-case Best: {save_best_dir}\n")
print(f"[Saved] summary written to: {summary_path}")
