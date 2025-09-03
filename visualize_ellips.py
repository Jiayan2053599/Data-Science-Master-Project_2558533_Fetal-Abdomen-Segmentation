# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm   # 新增：回归 + CI/PI

# =============== 路径配置（按需修改） ===============
image_dir = r"D:\TransUnet\usdata\val\image"  # 原图（所有帧在一个文件夹，文件名类似 0_472.png）
pred_mask_dir_unet      = r"D:\TransUnet\inference_results-UNet\binary\pred_masks"       # UNet 掩码（与原图同名）
pred_mask_dir_transunet = r"D:\TransUnet\inference_results-Transunet\binary\pred_masks"  # TransUNet 掩码（与原图同名）

# 映射表（你刚生成的）
case_to_uuid_csv = r"D:\TransUnet\case_to_uuid.csv"

# GT AC 的 CSV（uuid -> 多个 sweep 的 mm；默认取最大值）
gt_csv_path = r"D:\Data_Science_project-data\acouslic-ai-train-set\circumferences\fetal_abdominal_circumferences_per_sweep.csv"

# 像素→mm
SPACING_MM = 0.28

# 输出
out_csv      = r"D:\TransUnet\ac_compare_unet_transunet_vsCSV.csv"
triptych_dir = r"D:\TransUnet\triptych_topK"
TOPK = 20
os.makedirs(triptych_dir, exist_ok=True)

# 新增：标定图输出目录
calib_dir = os.path.join(os.path.dirname(out_csv), "calibration_plots")
os.makedirs(calib_dir, exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# =============== 读取映射与 GT ===============
map_df = pd.read_csv(case_to_uuid_csv)
case2uuid = dict(zip(map_df["case_id"].astype(str), map_df["uuid"].astype(str)))

gt_df = pd.read_csv(gt_csv_path, encoding="ISO-8859-1")
gt_df.columns = gt_df.columns.str.strip()
gt_df.rename(columns={gt_df.columns[0]:"uuid"}, inplace=True)
sweep_cols = [c for c in gt_df.columns if c.lower().startswith("sweep")]
gt_df["ac_gt_mm"] = gt_df[sweep_cols].max(axis=1)
uuid2gt = dict(zip(gt_df["uuid"].astype(str), gt_df["ac_gt_mm"].astype(float)))

# =============== 工具函数 ===============
def read_gray(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE)

def read_mask_bin(p):
    if p is None or not os.path.exists(p): return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    return (m >= 128).astype(np.uint8)

def compute_ac_from_mask(mask):
    if mask is None or mask.max()==0: return None, None
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, None
    largest = max(cnts, key=cv2.contourArea)
    if len(largest) < 5: return None, None
    ellipse = cv2.fitEllipse(largest)
    a = ellipse[1][0] / 2.0; b = ellipse[1][1] / 2.0
    ac_pixels = np.pi * (3*(a + b) - np.sqrt((3*a + b)*(a + 3*b)))
    return ac_pixels, ellipse

def overlay_pred(gray, mask_bool, color=(0,255,0), alpha=0.35):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    b,g,r = color
    for c,val in enumerate([b,g,r]):
        ch = rgb[...,c].astype(np.float32)
        ch[mask_bool] = (1-alpha)*ch[mask_bool] + alpha*val
        rgb[...,c] = ch.clip(0,255).astype(np.uint8)
    return rgb

def save_triptych(base_save, model_name, fname, gray, pred_mask, ellipse, ac_pred_mm, ac_gt_mm):
    if pred_mask is None: return
    if pred_mask.shape != gray.shape:
        pred_mask = cv2.resize(pred_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_bool = pred_mask.astype(bool)

    panel1 = gray
    panel2 = overlay_pred(gray, pred_bool, (0,255,0), 0.35)
    panel3 = overlay_pred(gray, pred_bool, (0,255,0), 0.35)
    if ellipse is not None:
        cv2.ellipse(panel3, ellipse, (0,165,255), 2)
    txt = f"GT: {ac_gt_mm:.1f} | {model_name}: {ac_pred_mm:.1f} | Err: {abs(ac_pred_mm-ac_gt_mm):.1f} mm"
    cv2.rectangle(panel3, (8, 8), (8+int(7*len(txt)), 36), (0,0,0), -1)
    cv2.putText(panel3, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    plt.figure(figsize=(12,4))
    for i,(im,title) in enumerate([(panel1,"Image"),
                                   (panel2[...,::-1],"Pred Overlay"),
                                   (panel3[...,::-1],"Ellipse + AC")],1):
        ax = plt.subplot(1,3,i); ax.axis('off')
        if im.ndim==2: ax.imshow(im, cmap='gray')
        else: ax.imshow(im)
        ax.set_title(title)
    out = os.path.join(base_save, f"{os.path.splitext(fname)[0]}_{model_name}_triptych.png")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    return out

def find_same_name(root, stem):
    for ext in IMG_EXTS:
        p = os.path.join(root, stem + ext)
        if os.path.exists(p): return p
    return None

# ---------- 新增：画 AC vs GT 的回归+区间（含 NAE 与保留率）----------
def plot_calibration_scatter(
    df_use, y_col_pred, model_name, save_path,
    title_suffix="",
    point_alpha=0.6, ci_alpha=0.20, pi_alpha=0.12,
    add_quartiles=False,          # 是否在角标里附带 AE/NAE 的中位数与四分位
    nae_eps=1e-6,                 # 防止除零
    total_N=None                  # 若传入“筛选前样本总数”，将显示保留率
):
    """
    df_use: DataFrame，至少包含列 ["ac_gt_mm", y_col_pred]，NaN/无效值已剔除
    y_col_pred: 例如 "ac_unet_mm" 或 "ac_transunet_mm"
    model_name: 图中 y 轴/标题里显示的模型名
    save_path:  保存路径（含文件名）
    """

    if df_use.shape[0] < 2:
        print(f"[Warn] {model_name} 样本数不足(<2)，跳过绘图：{title_suffix}")
        return

    # -------- 数据取出 --------
    x = df_use["ac_gt_mm"].to_numpy(float)      # GT
    y = df_use[y_col_pred].to_numpy(float)      # Pred

    # -------- 误差与 NAE --------
    err = y - x
    ae  = np.abs(err)
    denom = np.maximum(np.maximum(y, x), nae_eps)      # 与筛选口径一致
    nae = ae / denom

    # -------- OLS 拟合 --------
    X_ols = sm.add_constant(x)
    model_ols = sm.OLS(y, X_ols).fit()
    a, b = model_ols.params
    r2 = model_ols.rsquared

    # 生成等距横轴用于画线/区间
    x_grid = np.linspace(x.min(), x.max(), 200)
    Xg = sm.add_constant(x_grid)
    pred = model_ols.get_prediction(Xg)
    sf = pred.summary_frame(alpha=0.05)  # 95% CI/PI

    # -------- 统计量 --------
    mae  = float(ae.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())
    nae_mean = float(nae.mean())
    nae_std  = float(nae.std())

    if add_quartiles:
        ae_q1,  ae_med,  ae_q3  = np.percentile(ae,  [25, 50, 75])
        nae_q1, nae_med, nae_q3 = np.percentile(nae, [25, 50, 75])

    # -------- 绘图 --------
    plt.figure(figsize=(7.2, 6.0))
    plt.scatter(x, y, alpha=point_alpha, label="Samples")

    # 回归均值线 & CI
    plt.plot(x_grid, sf["mean"], 'r-', label="Trend (OLS)")
    plt.fill_between(x_grid, sf["mean_ci_lower"], sf["mean_ci_upper"],
                     color='r', alpha=ci_alpha, label="95% CI")
    # 预测区间（单样本波动范围）
    plt.fill_between(x_grid, sf["obs_ci_lower"], sf["obs_ci_upper"],
                     color='orange', alpha=pi_alpha, label="95% PI")

    # y=x 理想线
    ax_min = min(x.min(), y.min())
    ax_max = max(x.max(), y.max())
    plt.plot([ax_min, ax_max], [ax_min, ax_max], 'k--', label="Ideal y = x")

    plt.xlabel("Ground-Truth AC (mm)")
    plt.ylabel(f"Predicted AC (mm) — {model_name}")
    title_core = f"AC Prediction vs Ground Truth — {model_name}"
    if title_suffix:
        title_core += f" ({title_suffix})"
    plt.title(title_core)

    # 角标文本（统一包含 NAE）
    keep_info = ""
    if isinstance(total_N, (int, np.integer)) and total_N > 0:
        keep_info = f"  Retained={len(x)}/{total_N} ({len(x)/total_N:.1%})"

    txt = (f"Pred = {a:.3f} + {b:.3f}·GT   R²={r2:.3f}\n"
           f"N={len(x)}  MAE={mae:.1f}  RMSE={rmse:.1f}  Bias={bias:.1f}  "
           f"NAE={nae_mean:.3f}±{nae_std:.3f}{keep_info}")

    if add_quartiles:
        txt += (f"\nAE med [Q1,Q3]={ae_med:.1f} [{ae_q1:.1f},{ae_q3:.1f}] mm   "
                f"NAE med [Q1,Q3]={nae_med:.3f} [{nae_q1:.3f},{nae_q3:.3f}]")

    plt.gca().text(0.02, 0.02, txt, transform=plt.gca().transAxes,
                   ha='left', va='bottom', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 控制台日志
    print(f"[Saved] {model_name} calibration plot -> {save_path}")
    print(f"  OLS: Pred = {a:.3f} + {b:.3f} · GT | R^2={r2:.3f} | "
          f"N={len(x)} MAE={mae:.2f} RMSE={rmse:.2f} Bias={bias:.2f} "
          f"NAE={nae_mean:.3f}±{nae_std:.3f}{keep_info}")

    # 返回统计结果，便于外部保存
    return {
        "model": model_name,
        "N": int(len(x)),
        "a": float(a), "b": float(b), "R2": float(r2),
        "MAE": mae, "RMSE": rmse, "Bias": bias,
        "NAE_mean": nae_mean, "NAE_std": nae_std,
        "AE_median": float(np.median(ae)),
        "NAE_median": float(np.median(nae)),
        "AE_Q1": float(np.percentile(ae, 25)),
        "AE_Q3": float(np.percentile(ae, 75)),
        "NAE_Q1": float(np.percentile(nae, 25)),
        "NAE_Q3": float(np.percentile(nae, 75)),
        "retained": (int(len(x)), int(total_N)) if total_N else None
    }

# =============== 主流程（逐帧） ===============
img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXTS)])
rows = []

for fname in tqdm(img_files, desc="AC via UNet & TransUNet (GT from CSV+mapping)"):
    stem = os.path.splitext(fname)[0]     # e.g., 0_472
    case_id = stem.split('_', 1)[0]       # '0'
    uuid = case2uuid.get(case_id)         # 由 case_to_uuid.csv 提供
    ac_gt_mm = uuid2gt.get(uuid)          # GT 来自 CSV

    gray = read_gray(os.path.join(image_dir, fname))
    if gray is None: continue

    # 掩码
    m_unet_path  = find_same_name(pred_mask_dir_unet, stem)
    m_tran_path  = find_same_name(pred_mask_dir_transunet, stem)
    m_unet  = read_mask_bin(m_unet_path)
    m_tran  = read_mask_bin(m_tran_path)

    # UNet
    ac_unet_mm = err_unet = nae_unet = None; ell_u = None
    if m_unet is not None:
        if m_unet.shape != gray.shape:
            m_unet = cv2.resize(m_unet, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        ac_pix, ell_u = compute_ac_from_mask(m_unet)
        if ac_pix is not None:
            ac_unet_mm = ac_pix * SPACING_MM
            if ac_gt_mm is not None and np.isfinite(ac_gt_mm):
                err_unet = abs(ac_unet_mm - ac_gt_mm)
                nae_unet = err_unet / max(ac_unet_mm, ac_gt_mm)

    # TransUNet
    ac_tran_mm = err_tran = nae_tran = None; ell_t = None
    if m_tran is not None:
        if m_tran.shape != gray.shape:
            m_tran = cv2.resize(m_tran, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        ac_pix, ell_t = compute_ac_from_mask(m_tran)
        if ac_pix is not None:
            ac_tran_mm = ac_pix * SPACING_MM
            if ac_gt_mm is not None and np.isfinite(ac_gt_mm):
                err_tran = abs(ac_tran_mm - ac_gt_mm)
                nae_tran = err_tran / max(ac_tran_mm, ac_gt_mm)

    rows.append({
        "filename": fname,
        "case_id": case_id,
        "uuid": uuid,
        "ac_gt_mm": ac_gt_mm,
        "ac_unet_mm": ac_unet_mm,
        "ac_transunet_mm": ac_tran_mm,
        "abs_error_unet": err_unet,
        "abs_error_transunet": err_tran,
        "nae_unet": nae_unet,
        "nae_transunet": nae_tran
    })

# =============== 保存 CSV ===============
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"[Saved] results -> {out_csv}")

# =============== Top-K 三联图（按误差） ===============
if df["ac_gt_mm"].notna().any():
    # UNet
    base_u = os.path.join(triptych_dir, "UNet"); os.makedirs(base_u, exist_ok=True)
    top_u = df.dropna(subset=["abs_error_unet"]).sort_values("abs_error_unet").head(TOPK)
    for _, r in top_u.iterrows():
        gray = read_gray(os.path.join(image_dir, r["filename"]))
        m = read_mask_bin(find_same_name(pred_mask_dir_unet, os.path.splitext(r["filename"])[0]))
        _, ell = compute_ac_from_mask(m) if m is not None else (None, None)
        if (r["ac_unet_mm"] is not None) and (r["ac_gt_mm"] is not None):
            save_triptych(base_u, "UNet", r["filename"], gray, m, ell, r["ac_unet_mm"], r["ac_gt_mm"])
    print(f"[Saved] UNet top-{TOPK} triptychs -> {base_u}")

    # TransUNet
    base_t = os.path.join(triptych_dir, "TransUNet"); os.makedirs(base_t, exist_ok=True)
    top_t = df.dropna(subset=["abs_error_transunet"]).sort_values("abs_error_transunet").head(TOPK)
    for _, r in top_t.iterrows():
        gray = read_gray(os.path.join(image_dir, r["filename"]))
        m = read_mask_bin(find_same_name(pred_mask_dir_transunet, os.path.splitext(r["filename"])[0]))
        _, ell = compute_ac_from_mask(m) if m is not None else (None, None)
        if (r["ac_transunet_mm"] is not None) and (r["ac_gt_mm"] is not None):
            save_triptych(base_t, "TransUNet", r["filename"], gray, m, ell, r["ac_transunet_mm"], r["ac_gt_mm"])
    print(f"[Saved] TransUNet top-{TOPK} triptychs -> {base_t}")
else:
    print("映射或文件名未能关联到任何 uuid（ac_gt_mm 全空）。请检查 case_id 是否与 PNG 前缀一致、映射是否与导出顺序同源。")

# =============== 新增：分别可视化“AC 预测 vs. GT”（两模型、全量 & 过滤后） ===============
# 过滤阈值（可按需调整）
ABS_ERR_THR = 60.0   # mm
NAE_THR     = 0.30   # 30%

# ---- UNet ----
df_unet_all = df[["ac_gt_mm", "ac_unet_mm"]].dropna()
if len(df_unet_all) >= 2:
    save_all = os.path.join(calib_dir, "ac_vs_gt_UNet_all.png")
    plot_calibration_scatter(df_unet_all, "ac_unet_mm", "UNet", save_all, title_suffix="All Data")

    df_unet_f = df.dropna(subset=["ac_gt_mm","ac_unet_mm","abs_error_unet","nae_unet"])
    df_unet_f = df_unet_f[(df_unet_f["abs_error_unet"] <= ABS_ERR_THR) & (df_unet_f["nae_unet"] <= NAE_THR)][["ac_gt_mm","ac_unet_mm"]]
    if len(df_unet_f) >= 2:
        save_f = os.path.join(calib_dir, "ac_vs_gt_UNet_filtered.png")
        plot_calibration_scatter(df_unet_f, "ac_unet_mm", "UNet", save_f, title_suffix=f"Filtered (|err|≤{ABS_ERR_THR}mm & NAE≤{NAE_THR*100:.0f}%)")
    else:
        print("[Warn] UNet 过滤后样本不足(<2)，跳过过滤图。")
else:
    print("[Warn] UNet 有效样本不足(<2)，跳过回归图。")

# ---- TransUNet ----
df_tran_all = df[["ac_gt_mm", "ac_transunet_mm"]].dropna()
if len(df_tran_all) >= 2:
    save_all = os.path.join(calib_dir, "ac_vs_gt_TransUNet_all.png")
    plot_calibration_scatter(df_tran_all, "ac_transunet_mm", "TransUNet", save_all, title_suffix="All Data")

    df_tran_f = df.dropna(subset=["ac_gt_mm","ac_transunet_mm","abs_error_transunet","nae_transunet"])
    df_tran_f = df_tran_f[(df_tran_f["abs_error_transunet"] <= ABS_ERR_THR) & (df_tran_f["nae_transunet"] <= NAE_THR)][["ac_gt_mm","ac_transunet_mm"]]
    if len(df_tran_f) >= 2:
        save_f = os.path.join(calib_dir, "ac_vs_gt_TransUNet_filtered.png")
        plot_calibration_scatter(df_tran_f, "ac_transunet_mm", "TransUNet", save_f, title_suffix=f"Filtered (|err|≤{ABS_ERR_THR}mm & NAE≤{NAE_THR*100:.0f}%)")
    else:
        print("[Warn] TransUNet 过滤后样本不足(<2)，跳过过滤图。")
else:
    print("[Warn] TransUNet 有效样本不足(<2)，跳过回归图。")

print(f"[Done] Calibration figures saved under: {calib_dir}")
