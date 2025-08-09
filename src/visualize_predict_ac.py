"""

Created on 2025/8/2 03:22
@author: 18310

"""

"""
Created on 2025/8/2 03:22
@author: 18310
"""

import os
import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# ========== 路径设置 ==========
pred_mask_dir = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_aplit_correct)\infer_output"
save_vis_dir = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_aplit_correct)\ac_ellipse_vis"
os.makedirs(save_vis_dir, exist_ok=True)

# ========== 椭圆拟合函数 ==========
def compute_ellipse(mask_2d):
    contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    return cv2.fitEllipse(largest)

# ========== 可视化函数 ==========
def visualize_prediction_with_ellipse(mask_path):
    fname = os.path.basename(mask_path)
    image = sitk.ReadImage(mask_path)

    # 按 challenge 标准固定 spacing
    spacing_mm = 0.28

    mask_np = sitk.GetArrayFromImage(image)
    mask_2d = mask_np[0] if mask_np.ndim == 3 else mask_np
    mask_gray = (mask_2d * 255).astype(np.uint8)

    ellipse = compute_ellipse(mask_2d)
    if ellipse is None:
        print(f"[跳过] {fname} 无法拟合椭圆")
        return

    a = ellipse[1][0] / 2
    b = ellipse[1][1] / 2
    ac_pixels = np.pi * (3*(a + b) - np.sqrt((3*a + b)*(a + 3*b)))
    ac_mm = ac_pixels * spacing_mm

    canvas = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(canvas, ellipse, (0, 255, 0), 2)
    cv2.putText(canvas, f"AC: {ac_mm:.1f} mm", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(canvas[..., ::-1])
    plt.title(fname.replace(".nii.gz", ""))
    plt.axis("off")
    plt.tight_layout()
    out_path = os.path.join(save_vis_dir, fname.replace(".nii.gz", "_ellipse.png"))
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close()


# ========== 主流程 ==========
all_preds = glob(os.path.join(pred_mask_dir, "*.nii.gz"))
for pred_path in tqdm(all_preds, desc="可视化中"):
    visualize_prediction_with_ellipse(pred_path)

print(f"\n可视化完成，共处理 {len(all_preds)} 个样本\n结果已保存至: {save_vis_dir}")

