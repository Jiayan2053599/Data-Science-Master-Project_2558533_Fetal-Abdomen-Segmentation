# -*- coding: utf-8 -*-
"""
TransUNet inference (aligned)
- Binary: All-FG (p1+p2>0.5)
- Multi : merge-FG morphology (close→fill holes→rm small CC), then restore classes
- Plane-sanity (multi): majority/minority ratio, optional enforcement to avoid mixing
- Primary metric: per-case, All-FG
- Visualization:
    (1) Top-K triptychs across slices
    (2) Per-case best slice triptych (one figure per case)
"""
import os, sys, cv2, torch, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms
from scipy.ndimage import binary_fill_holes

# ======================= Config =======================
IS_BINARY_MODE        = False     # True=Binary / False=Multi
BINARY_POSTPROCESS    = True
ENABLE_VISUALIZATION  = True
TOPK_TRIPTYCHS        = 20        # 全局TopK（跨病例）
SAVE_PER_CASE_BEST    = True      # 每病例仅保存最佳一帧
FG_THRESH             = 0.5

# ---- Plane consistency (only for multi-class) ----
PLANE_SANITY              = True      # 是否统计/执行平面一致化
PLANE_SANITY_SCOPE        = "lcc"     # "all_fg" 或 "lcc"（仅最大连通域上统计/修复）
PLANE_SANITY_ENFORCE      = True      # True=把前景强制改成多数类
PLANE_ENFORCE_MINORITY_GT = 0.00      # 当 minority_share > 该阈值时才执行强制（0 表示只要存在混合就修复）
ANNOTATE_PLANE_INFO       = True      # 中间面板角标显示 Majority & mix

SHOW_TRIPTYCH_INLINE  = False     # 交互显示（一般关闭）
INLINE_SHOW_MAX       = 10
INLINE_SHOW_SECONDS   = 1.5

# ====== 路径 ======
val_img_dir   = r"D:\TransUnet\usdata\val\image"
val_mask_dir  = r"D:\TransUnet\usdata\val\mask"
model_path    = r"D:\TransUnet\trainingrecords_transUnet\checkpoint\mydata_transunet_Lovasz\mydata_transunet_Lovasz_0.5189_21.pkl"
base_save_dir = r"D:\TransUnet\inference_results-Transunet"

mode_folder      = 'binary' if IS_BINARY_MODE else 'multi'
save_pred_dir    = os.path.join(base_save_dir, mode_folder, 'pred_masks')
save_trip_dir    = os.path.join(base_save_dir, mode_folder, 'target_vis_comparison_topK')
save_best_dir    = os.path.join(base_save_dir, mode_folder, 'per_case_best')
save_csv_path    = os.path.join(base_save_dir, mode_folder, 'metrics_per_slice.csv')
os.makedirs(save_pred_dir, exist_ok=True)
os.makedirs(save_trip_dir, exist_ok=True)
os.makedirs(save_best_dir, exist_ok=True)

# =================== Matplotlib ===================
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ======================= Colors & names =======================
CLASS1_NAME = "Best Plane"
CLASS2_NAME = "Sub-Best Plane"

# --- BGR for OpenCV; legend uses RGB via helpers ---
COLORS = {
    # Binary (All-FG)
    "tp":  (0, 255,   0),   # 绿  TP
    "fn":  (0,   0, 255),   # 红  FN
    "fp":  (255, 0,   0),   # 蓝  FP

    # Multi-class (Best / Sub-best)
    "tp1": (0, 255,   0),   # 绿  TP1 - Best Plane Correct
    "fn1": (0,   0, 255),   # 红  FN1 - Best Plane Missed
    "fp1": (255, 0,   0),   # 蓝  FP1 - Best Plane False
    "tp2": (0, 255, 255),   # 黄  TP2 - Sub-Best Correct   (RGB 255,255,0)
    "fn2": (0, 165, 255),   # 橙  FN2 - Sub-Best Missed   (RGB 255,165,0)
    "fp2": (128, 0, 128),   # 紫  FP2 - Sub-Best False    (RGB 128,0,128)
}
# 中间图 class-2 使用“金黄”更柔和：RGB(255,215,0)
CLASS2_GOLD_RGB = (255, 215, 0)

ALPHA_TP  = 0.35   # TP 透明度
ALPHA_ERR = 0.60   # 误分类透明度

def _rgb(bgr):
    b,g,r = bgr
    return (r/255., g/255., b/255.)

def _rgb255(bgr):
    b,g,r = bgr
    return (r, g, b)

# ======================= Model =======================
sys.path.append(r"D:\TransUnet\ussegcodes")
from models.transunet import TransUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransUNet(img_dim=512, in_channels=1, out_channels=64,
                  head_num=4, mlp_dim=512, block_num=8, patch_dim=16,
                  class_num=3).to(device)

def _strip_module(sd):
    if len(sd) and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k,v in sd.items()}
    return sd

try:
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
except TypeError:
    ckpt = torch.load(model_path, map_location=device)

loaded = False
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(_strip_module(ckpt["model_state_dict"]), strict=True); loaded=True
elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    model.load_state_dict(_strip_module(ckpt), strict=True); loaded=True
else:
    for v in (ckpt.values() if isinstance(ckpt, dict) else []):
        if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
            try:
                model.load_state_dict(_strip_module(v), strict=True); loaded=True; break
            except:
                pass
if not loaded: raise ValueError("未找到可用的 state_dict，请检查权重。")
model.eval()

# ======================= Metrics =======================
def dice_score(pred, target, smooth=1e-6):
    p = (pred>0).astype(np.uint8); g = (target>0).astype(np.uint8)
    inter = (p & g).sum()
    return (2*inter + smooth) / (p.sum() + g.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    p = (pred>0).astype(np.uint8); g = (target>0).astype(np.uint8)
    inter = (p & g).sum(); union = (p | g).sum()
    return (inter + smooth) / (union + smooth)

# ======================= Morphology (for Multi FG) =======================
def postprocess_mask(bin_mask, ksize=5, area_thresh=100):
    kernel = np.ones((ksize, ksize), np.uint8)
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    filled = (binary_fill_holes(closed>0).astype(np.uint8)) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats((filled>0).astype(np.uint8), 8)
    clean = np.zeros_like(filled, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= area_thresh:
            clean[labels==i] = 255
    return clean

# ======================= Plane sanity (multi) =======================
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

# ======================= Triptych helpers =======================
def _ensure_rgb_from_gray(gray):
    if gray.ndim == 2:
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        base = gray.copy()
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return base

def _alpha_blend(base_rgb, mask_bool, color_rgb255, alpha):
    if not mask_bool.any(): return
    r,g,b = color_rgb255
    base_rgb[..., 0][mask_bool] = (1-alpha)*base_rgb[..., 0][mask_bool] + alpha*r
    base_rgb[..., 1][mask_bool] = (1-alpha)*base_rgb[..., 1][mask_bool] + alpha*g
    base_rgb[..., 2][mask_bool] = (1-alpha)*base_rgb[..., 2][mask_bool] + alpha*b

def build_pred_overlay_rgb(gray, pred_map, binary_mode=False):
    """
    中间图：预测叠加（互斥上色）
      - binary: 预测前景统一绿色
      - multi : 1=绿（Best），2=金黄（Sub-best）
    """
    base = _ensure_rgb_from_gray(gray)
    if binary_mode:
        pr = (pred_map > 0)
        _alpha_blend(base, pr, _rgb255(COLORS['tp']), ALPHA_TP)
    else:
        pr1 = (pred_map == 1)
        pr2 = (pred_map == 2)
        _alpha_blend(base, pr1, _rgb255(COLORS['tp1']), ALPHA_TP)      # 绿
        _alpha_blend(base, pr2, CLASS2_GOLD_RGB, ALPHA_TP)             # 金黄
    return np.clip(base, 0, 255).astype(np.uint8)

def build_confusion_overlay_rgb(gray, gt_map, pred_map, binary_mode=False):
    """
    右图：TP/FN/FP 互斥六色/三色
    """
    base = _ensure_rgb_from_gray(gray)
    if binary_mode:
        gt = (gt_map>0); pr = (pred_map>0)
        tp = gt & pr; fn = gt & (~pr); fp = (~gt) & pr
        _alpha_blend(base, tp, _rgb255(COLORS['tp']), ALPHA_TP)
        _alpha_blend(base, fn, _rgb255(COLORS['fn']), ALPHA_ERR)
        _alpha_blend(base, fp, _rgb255(COLORS['fp']), ALPHA_ERR)
    else:
        gt1 = (gt_map==1); gt2=(gt_map==2)
        pr1 = (pred_map==1); pr2=(pred_map==2)
        tp1 = gt1 & pr1; fn1 = gt1 & (~pr1); fp1 = (~gt1) & pr1
        tp2 = gt2 & pr2; fn2 = gt2 & (~pr2); fp2 = (~gt2) & pr2

        # 先画错，再画 TP（不叠色）
        for m, key, is_err in [
            (fn1,'fn1',True),(fp1,'fp1',True),
            (fn2,'fn2',True),(fp2,'fp2',True),
            (tp1,'tp1',False),(tp2,'tp2',False)
        ]:
            _alpha_blend(base, m, _rgb255(COLORS[key]), ALPHA_ERR if is_err else ALPHA_TP)

    return np.clip(base, 0, 255).astype(np.uint8)

def render_triptych_figure(gray, gt_map, pred_map, binary_mode=False, plane_info=None):
    """
    三联图：左 原图；中 预测叠加；右 误差图（带图例）
    plane_info: (majority_plane:int, minority_share:float) 或 None
    """
    left_img = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    mid_rgb  = build_pred_overlay_rgb(gray, pred_map, binary_mode=binary_mode)
    overlay_rgb = build_confusion_overlay_rgb(gray, gt_map, pred_map, binary_mode=binary_mode)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax in axes: ax.axis('off')

    axes[0].imshow(left_img, cmap=('gray' if left_img.ndim==2 else None))
    axes[0].set_title('Image')

    axes[1].imshow(mid_rgb)
    axes[1].set_title('Pred Overlay (Best/Sub-best)' if not binary_mode else 'Pred Overlay (Merged)')

    # 角标：多数类与混合比例
    if (not binary_mode) and ANNOTATE_PLANE_INFO and plane_info is not None:
        maj, mix = plane_info
        axes[1].text(0.02, 0.98, f"Majority={maj}  mix={mix:.3f}",
                     transform=axes[1].transAxes, va='top', ha='left',
                     fontsize=11, color='white',
                     bbox=dict(facecolor='black', alpha=0.5, pad=3, edgecolor='none'))

    axes[2].imshow(overlay_rgb)
    axes[2].set_title('Overlay (Error Map)')

    if binary_mode:
        legend_patches = [
            Patch(color=_rgb(COLORS['tp']), label="TP (Correct FG)"),
            Patch(color=_rgb(COLORS['fn']), label="FN (Missed FG)"),
            Patch(color=_rgb(COLORS['fp']), label="FP (False FG)")
        ]
    else:
        legend_patches = [
            Patch(color=_rgb(COLORS['tp1']), label="TP1 - Best Plane Correct"),
            Patch(color=_rgb(COLORS['fn1']), label="FN1 - Best Plane Missed"),
            Patch(color=_rgb(COLORS['fp1']), label="FP1 - Best Plane False"),
            Patch(color=_rgb(COLORS['tp2']), label="TP2 - Sub-Best Correct"),
            Patch(color=_rgb(COLORS['fn2']), label="FN2 - Sub-Best Missed"),
            Patch(color=_rgb(COLORS['fp2']), label="FP2 - Sub-Best False"),
        ]
    leg = axes[2].legend(handles=legend_patches, loc='lower right', fontsize=9, framealpha=0.85)
    for lh in getattr(leg, "legendHandles", []):
        try: lh.set_alpha(1.0)
        except: pass

    plt.tight_layout()
    return fig

# ======================= Inference =======================
transform = transforms.ToTensor()
def get_case_id(fname: str) -> str:
    # 以第一个下划线前作为病例ID
    return os.path.splitext(fname)[0].split('_')[0]

rows_slicewise = []
case_scores = {}
case_scores_c1, case_scores_c2 = {}, {}
neg_total_slices = 0
neg_fp_slices    = 0

# 每病例最佳帧缓存：case_id -> (best_dice, record_tuple)
# record_tuple: (dice, case_id, fname, gray, gt_map, pred_map, is_binary, plane_info)
best_trip_by_case = {}

# 仍保留跨病例Top-K池
trip_pool = []
rank_idx  = 0

file_names = sorted([f for f in os.listdir(val_img_dir)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))])

model.eval()
for fname in tqdm(file_names, desc="Inferencing (TransUNet, aligned)"):
    img_path = os.path.join(val_img_dir, fname)
    msk_path = os.path.join(val_mask_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 原图灰度
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        print(f"[Skip] {fname}"); continue

    case_id = get_case_id(fname)
    case_scores.setdefault(case_id, [])
    case_scores_c1.setdefault(case_id, [])
    case_scores_c2.setdefault(case_id, [])

    with torch.no_grad():
        logits = model(transform(img).unsqueeze(0).to(device))
        if logits.ndim == 3: logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()  # (C,H,W)

    if IS_BINARY_MODE:
        fg_prob  = (probs[1] if probs.shape[0]>1 else 0) + (probs[2] if probs.shape[0]>2 else 0)
        pred_bin = (fg_prob > FG_THRESH).astype(np.uint8) * 255
        if BINARY_POSTPROCESS:
            pred_bin = postprocess_mask(pred_bin)
        gt_bin   = (msk > 0).astype(np.uint8) * 255

        d_sw = dice_score(pred_bin, gt_bin)
        i_sw = iou_score(pred_bin,  gt_bin)
        # 二类：无 plane 字段
        rows_slicewise.append([fname, d_sw, i_sw, np.nan, np.nan])

        if gt_bin.sum() > 0:
            case_scores[case_id].append((d_sw, i_sw))
        else:
            neg_total_slices += 1
            if pred_bin.sum() > 0: neg_fp_slices += 1

        cv2.imwrite(os.path.join(save_pred_dir, fname), pred_bin)

        if ENABLE_VISUALIZATION:
            rec = (float(d_sw), case_id, fname, img, gt_bin, pred_bin, True, None)
            trip_pool.append((float(d_sw), rank_idx, rec)); rank_idx += 1
            if (case_id not in best_trip_by_case) or (d_sw > best_trip_by_case[case_id][0]):
                best_trip_by_case[case_id] = (float(d_sw), rec)

    else:
        # 原始 argmax 类别
        am_raw = np.argmax(probs, axis=0).astype(np.uint8)   # 0/1/2
        # 合并前景做形态学
        merged = postprocess_mask(((am_raw>0).astype(np.uint8) * 255))
        am = np.zeros_like(am_raw, dtype=np.uint8)
        am[(merged>0) & (am_raw==1)] = 1
        am[(merged>0) & (am_raw==2)] = 2

        # ★ 平面一致化：统计/（可选）修复
        maj_plane, mix_ratio = np.nan, np.nan
        if PLANE_SANITY:
            am, maj_plane, mix_ratio = apply_plane_sanity(
                am,
                scope=PLANE_SANITY_SCOPE,
                enforce=PLANE_SANITY_ENFORCE,
                enforce_thr=PLANE_ENFORCE_MINORITY_GT
            )

        pred_bin = (am>0).astype(np.uint8) * 255
        gt_bin   = (msk>0).astype(np.uint8) * 255

        d_sw = dice_score(pred_bin, gt_bin)     # 主评估用 All-FG
        i_sw = iou_score(pred_bin,  gt_bin)
        rows_slicewise.append([fname, d_sw, i_sw, (maj_plane if not np.isnan(maj_plane) else np.nan),
                               (mix_ratio if not np.isnan(mix_ratio) else np.nan)])

        if gt_bin.sum() > 0:
            case_scores[case_id].append((d_sw, i_sw))
        else:
            neg_total_slices += 1
            if pred_bin.sum() > 0: neg_fp_slices += 1

        # 记录每类Dice（可选）
        gt1 = (msk==100).astype(np.uint8)*255; gt2 = (msk==255).astype(np.uint8)*255
        pr1 = (am==1).astype(np.uint8)*255;   pr2 = (am==2).astype(np.uint8)*255
        if gt1.sum()>0:
            case_scores_c1[case_id].append((dice_score(pr1,gt1), iou_score(pr1,gt1)))
        if gt2.sum()>0:
            case_scores_c2[case_id].append((dice_score(pr2,gt2), iou_score(pr2,gt2)))

        # 保存预测mask（100/255编码）
        pred_mask = np.zeros_like(am, dtype=np.uint8)
        pred_mask[am==1] = 100; pred_mask[am==2] = 255
        cv2.imwrite(os.path.join(save_pred_dir, fname), pred_mask)

        if ENABLE_VISUALIZATION:
            gt_multi = np.zeros_like(am, dtype=np.uint8)
            gt_multi[msk==100] = 1; gt_multi[msk==255] = 2
            plane_info = (int(maj_plane) if not np.isnan(maj_plane) else None,
                          float(mix_ratio) if not np.isnan(mix_ratio) else None)
            rec = (float(d_sw), case_id, fname, img, gt_multi, am, False, plane_info)
            trip_pool.append((float(d_sw), rank_idx, rec)); rank_idx += 1
            if (case_id not in best_trip_by_case) or (d_sw > best_trip_by_case[case_id][0]):
                best_trip_by_case[case_id] = (float(d_sw), rec)

# ======================= Save Top-K Triptychs =======================
if ENABLE_VISUALIZATION and len(trip_pool)>0 and TOPK_TRIPTYCHS>0:
    shown = 0
    trip_pool.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for rank, (_, _, rec) in enumerate(trip_pool[:TOPK_TRIPTYCHS], start=1):
        d, case_id, fname, gray, gt_map, pred_map, is_binary, plane_info = rec
        fig = render_triptych_figure(gray, gt_map, pred_map, binary_mode=is_binary, plane_info=plane_info)
        outp = os.path.join(save_trip_dir, f"{rank:02d}_{d:.4f}_{case_id}_{os.path.splitext(fname)[0]}_triptych.png")
        fig.savefig(outp, dpi=200, bbox_inches='tight')
        if SHOW_TRIPTYCH_INLINE and shown < INLINE_SHOW_MAX and matplotlib.get_backend().lower()!='agg':
            plt.show(block=False); plt.pause(INLINE_SHOW_SECONDS); shown += 1
        plt.close(fig)

# ======================= Save Per-Case Best Triptychs =======================
if ENABLE_VISUALIZATION and SAVE_PER_CASE_BEST and len(best_trip_by_case) > 0:
    per_case_rows = []
    for case_id, (best_d, rec) in sorted(best_trip_by_case.items()):
        d, _, fname, gray, gt_map, pred_map, is_binary, plane_info = rec
        fig = render_triptych_figure(gray, gt_map, pred_map, binary_mode=is_binary, plane_info=plane_info)
        outp = os.path.join(save_best_dir, f"{case_id}_best_{d:.4f}_{os.path.splitext(fname)[0]}_triptych.png")
        fig.savefig(outp, dpi=200, bbox_inches='tight'); plt.close(fig)
        per_case_rows.append([case_id, fname, d])
    pd.DataFrame(per_case_rows, columns=["case_id","best_frame","best_dice_allFG"]).to_csv(
        os.path.join(base_save_dir, mode_folder, "per_case_best_list.csv"), index=False
    )

# ======================= Aggregate & report =======================
df = pd.DataFrame(rows_slicewise,
                  columns=["filename","dice_allFG","iou_allFG","majority_plane","minority_share"])
df.to_csv(save_csv_path, index=False)

def _percase_stats(bucket: dict):
    case_d, case_i = [], []
    for _case, lst in bucket.items():
        if len(lst)==0: continue
        case_d.append(float(np.mean([d for d,_ in lst])))
        case_i.append(float(np.mean([i for _,i in lst])))
    n = len(case_d)
    mean_d = float(np.mean(case_d)) if n else float('nan')
    mean_i = float(np.mean(case_i)) if n else float('nan')
    sd_d   = float(np.std(case_d, ddof=1)) if n>1 else float('nan')
    sd_i   = float(np.std(case_i, ddof=1)) if n>1 else float('nan')
    return mean_d, mean_i, n, sd_d, sd_i

mean_dice_pc, mean_iou_pc, n_cases, sd_dice_pc, sd_iou_pc = _percase_stats(case_scores)
if not IS_BINARY_MODE:
    mean_dice_c1, mean_iou_c1, n_c1, sd_dice_c1, sd_iou_c1 = _percase_stats(case_scores_c1)
    mean_dice_c2, mean_iou_c2, n_c2, sd_dice_c2, sd_iou_c2 = _percase_stats(case_scores_c2)
else:
    mean_dice_c1=mean_iou_c1=sd_dice_c1=sd_iou_c1=np.nan; n_c1=0
    mean_dice_c2=mean_iou_c2=sd_dice_c2=sd_iou_c2=np.nan; n_c2=0

neg_total = int(neg_total_slices)
neg_fpr   = float(neg_fp_slices/neg_total_slices) if neg_total_slices>0 else float('nan')

print("\n====== Evaluation (TransUNet) ======")
print(f"[Primary] (per-case, All-FG) Dice: {mean_dice_pc:.4f} ± {sd_dice_pc:.4f}  (N={n_cases})")
print(f"[Primary] (per-case, All-FG) IoU : {mean_iou_pc:.4f} ± {sd_iou_pc:.4f}  (N={n_cases})")
if not IS_BINARY_MODE:
    print(f"[Per-class] C1 Dice: {mean_dice_c1:.4f} ± {sd_dice_c1:.4f} | IoU: {mean_iou_c1:.4f} ± {sd_iou_c1:.4f} | cases: {n_c1}")
    print(f"[Per-class] C2 Dice: {mean_dice_c2:.4f} ± {sd_dice_c2:.4f} | IoU: {mean_iou_c2:.4f} ± {sd_iou_c2:.4f} | cases: {n_c2}")
print(f"Pred masks     -> {save_pred_dir}")
print(f"Triptychs TopK -> {save_trip_dir}")
print(f"Per-case Best  -> {save_best_dir}")
print(f"Per-slice CSV  -> {save_csv_path}")

# ======================= Summary file =======================
summary_path = os.path.join(base_save_dir, mode_folder, "metrics_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("===== TransUNet Inference Summary =====\n")
    f.write(f"Mode: {'Binary (All-FG)' if IS_BINARY_MODE else 'Multi-class (morph on merged FG)'}\n")
    f.write(f"Images dir: {val_img_dir}\n")
    f.write(f"Masks  dir: {val_mask_dir}\n")
    f.write(f"Model ckpt: {model_path}\n\n")
    f.write(f"[Primary] (per-case, All-FG) Dice: {mean_dice_pc:.6f} ± {sd_dice_pc:.6f} (N={n_cases})\n")
    f.write(f"[Primary] (per-case, All-FG) IoU : {mean_iou_pc:.6f} ± {sd_iou_pc:.6f} (N={n_cases})\n")
    if not IS_BINARY_MODE:
        f.write(f"[Per-class] C1 Dice: {mean_dice_c1:.6f} ± {sd_dice_c1:.6f} | IoU: {mean_iou_c1:.6f} ± {sd_iou_c1:.6f} | cases: {n_c1}\n")
        f.write(f"[Per-class] C2 Dice: {mean_dice_c2:.6f} ± {sd_dice_c2:.6f} | IoU: {mean_iou_c2:.6f} ± {sd_iou_c2:.6f} | cases: {n_c2}\n")
    f.write(f"Per-slice CSV: {save_csv_path}\n")
    f.write(f"Triptychs(top{TOPK_TRIPTYCHS}): {save_trip_dir}\n")
    f.write(f"Per-case Best: {save_best_dir}\n")
print(f"[Saved] summary written to: {summary_path}")
