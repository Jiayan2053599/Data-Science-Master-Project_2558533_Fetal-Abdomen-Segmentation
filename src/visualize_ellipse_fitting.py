import os
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm

# === 参数设置 ===
pred_mask_dir = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_aplit_correct)\infer_output"
csv_path = r"D:\Data_Science_project-data\acouslic-ai-train-set\circumferences\fetal_abdominal_circumferences_per_sweep.csv"
output_csv = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_aplit_correct)\ac_prediction_vs_gt.csv"
vis_dir = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_aplit_correct)\ac_ellipse_top5"
os.makedirs(vis_dir, exist_ok=True)

# === 椭圆拟合计算 AC（单位：像素） ===
def compute_ac_from_mask(mask_2d):
    contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    ellipse = cv2.fitEllipse(largest)
    a = ellipse[1][0] / 2  # 半长轴
    b = ellipse[1][1] / 2  # 半短轴
    ac_pixels = np.pi * (3*(a + b) - np.sqrt((3*a + b)*(a + 3*b)))
    return ac_pixels, ellipse

# === 加载 GT CSV（使用 uuid 与 mask 文件匹配） ===
gt_df = pd.read_csv(csv_path, encoding='ISO-8859-1')
gt_df.columns = gt_df.columns.str.strip()
gt_df.rename(columns={gt_df.columns[0]: "uuid"}, inplace=True)
sweep_cols = [col for col in gt_df.columns if col.startswith("sweep_")]
gt_df["max_ac_mm"] = gt_df[sweep_cols].max(axis=1)
gt_dict = dict(zip(gt_df["uuid"], gt_df["max_ac_mm"]))

# === 遍历预测结果并评估 ===
results = []
for fpath in tqdm(glob(os.path.join(pred_mask_dir, "*.nii.gz"))):
    fname = os.path.basename(fpath)
    study_id = fname.replace(".nii.gz", "")  # 去掉扩展名匹配 UUID

    if study_id not in gt_dict:
        print(f"[跳过] 未在GT中找到 study_id: {study_id}")
        continue

    image = sitk.ReadImage(fpath)

    # 固定 spacing - 按挑战赛规定
    spacing_mm = 0.28

    mask = sitk.GetArrayFromImage(image)
    mask_2d = mask[0] if mask.ndim == 3 else mask

    result = compute_ac_from_mask(mask_2d)
    if result is None:
        print(f"[跳过] {study_id} 无法拟合椭圆")
        continue

    ac_pixels, ellipse = result
    ac_pred_mm = ac_pixels * spacing_mm
    ac_gt_mm = gt_dict[study_id]
    abs_error = abs(ac_pred_mm - ac_gt_mm)
    nae_ac = abs_error / max(ac_pred_mm, ac_gt_mm)

    results.append({
        "study_id": study_id,
        "mask_path": fpath,
        "spacing_mm": spacing_mm,
        "ac_pred_mm": ac_pred_mm,
        "ac_gt_mm": ac_gt_mm,
        "abs_error_mm": abs_error,
        "nae_ac": nae_ac,
        "ellipse": ellipse
    })

    # 每个匹配到 GT 的样本直接打印结果
    print(f"{study_id}: spacing = {spacing_mm}, pred_ac = {ac_pred_mm:.2f}mm, GT = {ac_gt_mm:.2f}mm")


# 保存评估结果表格
df = pd.DataFrame(results).drop(columns=["ellipse"])
df.to_csv(output_csv, index=False)
print(f"比较完成，共评估 {len(df)} 个样本，结果保存为: {output_csv}")

if len(df) == 0:
    print("未能评估任何样本，可能是 UUID 匹配失败或 mask 无有效椭圆")
else:
    # ================= 原始数据统计 =================
    print("\n===== 原始数据统计 =====")
    print(df.describe())

    # --- 原始散点图 + 置信区间 ---
    x = df["ac_gt_mm"].to_numpy()
    y = df["ac_pred_mm"].to_numpy()
    X_ols = sm.add_constant(x)
    model_ols = sm.OLS(y, X_ols).fit()

    x_pred = np.linspace(x.min(), x.max(), 200)
    X_pred_ols = sm.add_constant(x_pred)
    pred_ols = model_ols.get_prediction(X_pred_ols)
    pred_summary = pred_ols.summary_frame(alpha=0.05)  # 95% CI

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, alpha=0.6, label="Samples")
    plt.plot(x_pred, pred_summary["mean"], 'r-', label="Trend (OLS)")
    plt.fill_between(x_pred,
                     pred_summary["mean_ci_lower"],
                     pred_summary["mean_ci_upper"],
                     color='r', alpha=0.2, label="95% CI")
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', label="Ideal y = x")
    plt.xlabel("Ground-Truth AC (mm)")
    plt.ylabel("Predicted AC (mm)")
    plt.title("AC Prediction vs Ground Truth (All Data)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================= 剔除异常值 =================
    abs_error_thresh = 60  # mm
    nae_thresh = 0.3       # 30%
    df_filtered = df[(df["abs_error_mm"] <= abs_error_thresh) & (df["nae_ac"] <= nae_thresh)]

    print("\n===== 剔除异常值后数据统计 =====")
    print(f"剔除异常值后样本数: {len(df_filtered)} / {len(df)}")
    print(df_filtered.describe())

    # 保存剔除后的结果
    filtered_csv = output_csv.replace(".csv", "_filtered.csv")
    df_filtered.to_csv(filtered_csv, index=False)
    print(f"剔除后的结果已保存: {filtered_csv}")

    # --- 剔除后散点图 + 置信区间 ---
    x_f = df_filtered["ac_gt_mm"].to_numpy()
    y_f = df_filtered["ac_pred_mm"].to_numpy()
    X_f_ols = sm.add_constant(x_f)
    model_f_ols = sm.OLS(y_f, X_f_ols).fit()

    x_f_pred = np.linspace(x_f.min(), x_f.max(), 200)
    X_f_pred_ols = sm.add_constant(x_f_pred)
    pred_f_ols = model_f_ols.get_prediction(X_f_pred_ols)
    pred_f_summary = pred_f_ols.summary_frame(alpha=0.05)

    plt.figure(figsize=(7, 6))
    plt.scatter(x_f, y_f, alpha=0.6, label="Samples")
    plt.plot(x_f_pred, pred_f_summary["mean"], 'r-', label="Trend (OLS)")
    plt.fill_between(x_f_pred,
                     pred_f_summary["mean_ci_lower"],
                     pred_f_summary["mean_ci_upper"],
                     color='r', alpha=0.2, label="95% CI")
    plt.plot([x_f.min(), x_f.max()], [x_f.min(), x_f.max()], 'k--', label="Ideal y = x")
    plt.xlabel("Ground-Truth AC (mm)")
    plt.ylabel("Predicted AC (mm)")
    plt.title("AC Prediction vs Ground Truth (Filtered)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ================= 误差最小的前 5 张图 =================
    top5 = sorted(results, key=lambda x: x['abs_error_mm'])[:5]
    for i, r in enumerate(top5):
        mask_img = sitk.GetArrayFromImage(sitk.ReadImage(r["mask_path"]))
        mask_2d = mask_img[0] if mask_img.ndim == 3 else mask_img
        canvas = cv2.cvtColor((mask_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.ellipse(canvas, r["ellipse"], (0, 255, 0), 2)
        text = f"GT: {r['ac_gt_mm']:.1f}mm | Pred: {r['ac_pred_mm']:.1f}mm"
        cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        plt.figure(figsize=(5, 5))
        plt.imshow(canvas[..., ::-1])
        plt.title(f"{r['study_id']} (Error: {r['abs_error_mm']:.1f}mm)")
        plt.axis("off")
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"{r['study_id']}_top{i+1}.png")
        plt.savefig(save_path)
        plt.show()
        plt.close()

    print(f"已保存误差最小的前 5 张预测图至：{vis_dir}")
# else:
#     print("未能评估任何样本，可能是 UUID 匹配失败或 mask 无有效椭圆")
