"""

Created on 2025/8/2 05:30
@author: 18310

"""

import os
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# === 路径设置 ===
pred_mask_dir = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(split_dataset_correct)\infer_output"
gt_mask_dir = r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal2D\labelsTs"
save_vis_dir = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(split_dataset_correct)\pred_vs_gt_vis"
os.makedirs(save_vis_dir, exist_ok=True)

# === 主函数 ===
def compare_masks(pred_path, gt_dir):
    fname = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, fname)  # 预测和GT名称应一致

    if not os.path.exists(gt_path):
        print(f"[跳过] 未找到 GT 文件: {gt_path}")
        return

    # 读取 mask（预测 & GT）
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    gt_img = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

    pred_2d = pred_img[0] if pred_img.ndim == 3 else pred_img
    gt_2d = gt_img[0] if gt_img.ndim == 3 else gt_img

    # 转为灰度图用于画布
    canvas = cv2.cvtColor((pred_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # === 添加红色轮廓（GT）
    gt_contours, _ = cv2.findContours(gt_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in gt_contours:
        cv2.drawContours(canvas, [c], -1, (0, 0, 255), 2)  # 红色

    # === 添加绿色拟合椭圆（预测）
    pred_contours, _ = cv2.findContours(pred_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if pred_contours and len(pred_contours[0]) >= 5:
        ellipse = cv2.fitEllipse(max(pred_contours, key=cv2.contourArea))
        cv2.ellipse(canvas, ellipse, (0, 255, 0), 2)  # 绿色

    # === 添加图例（Legend）
    legend_canvas = canvas.copy()
    cv2.rectangle(legend_canvas, (10, 10), (240, 60), (0, 0, 0), -1)  # 黑色底
    cv2.putText(legend_canvas, "GT (expert): Red", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(legend_canvas, "Predicted: Green", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # === 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(legend_canvas[..., ::-1])
    plt.title(fname.replace(".nii.gz", ""))
    plt.axis("off")
    plt.tight_layout()
    save_path = os.path.join(save_vis_dir, fname.replace(".nii.gz", "_compare.png"))
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

# === 批量处理 ===
all_pred = glob(os.path.join(pred_mask_dir, "*.nii.gz"))
for p in tqdm(all_pred, desc="正在生成预测 vs 专家对比图（含图例）"):
    compare_masks(p, gt_mask_dir)

print(f"完成！对比图保存至：{save_vis_dir}")

