"""

Created on 2025/7/27 18:12
@author: 18310

"""

import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- 全量数据实验 ----------
SUMMARY_PATH = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_split_correct)\MyTrainer__nnUNetPlans__2d\fold_0\validation\summary.json"
TOP_K = 5  # 选择前几个Dice最好的case
SAVE_DIR = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_split_correct)\MyTrainer__nnUNetPlans__2d\fold_0\visualizations_best_cases"
os.makedirs(SAVE_DIR, exist_ok=True)

# # ----------- 2000例正样本实验 ----------------
# SUMMARY_PATH = r"D:\nnUNet\results\Dataset050_AC_mha2d_top2000\MyTrainer__nnUNetPlans__2d\fold_0\validation\summary.json"
# TOP_K = 5  # 选择前几个Dice最好的case
# SAVE_DIR = r"D:\nnUNet\results\Dataset050_AC_mha2d_top2000\MyTrainer__nnUNetPlans__2d\fold_0\visualizations_best_cases"
# os.makedirs(SAVE_DIR, exist_ok=True)
# ----------------------------
# # ---------- 云端路径配置 ----------
# SUMMARY_PATH = "/home/ec2-user/nnUNet/nnUNet_results/Dataset050_AC_mha2d_top2000/MyTrainer__nnUNetPlans__2d/fold_0/validation/summary.json"
# TOP_K = 5
# SAVE_DIR = "/home/ec2-user/nnUNet/nnUNet_results/Dataset050_AC_mha2d_top2000/MyTrainer__nnUNetPlans__2d/fold_0/visualizations_best_cases"
# os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 加载 summary 文件 ----------

# 加载 summary 文件
with open(SUMMARY_PATH, 'r') as f:
    summary = json.load(f)

# 按Dice排序，选择top K
cases = summary['metric_per_case']
sorted_cases = sorted(cases, key=lambda x: x['metrics']['1']['Dice'], reverse=True)[:TOP_K]


# ---------- 替换路径为本地 ----------
for case in sorted_cases:
    case["prediction_file"] = case["prediction_file"].replace(
        "/home/ec2-user", "D:/"
    )
    case["reference_file"] = case["reference_file"].replace(
        "/home/ec2-user", "D:/"
    )
 # ✅ 补充这一步，修正路径中 Dataset 名称 (optional)
    case["prediction_file"] = case["prediction_file"].replace(
        "Dataset300_ACOptimalSuboptimal2D/", "Dataset300_ACOptimalSuboptimal2D(data_split_correct)/"
    )
    case["reference_file"] = case["reference_file"].replace(
        "Dataset300_ACOptimalSuboptimal2D/", "Dataset300_ACOptimalSuboptimal2D(data_split_correct)/"
    )


def visualize_case_multilabel(pred_path, gt_path, save_path, slice_idx=None):
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    assert pred.shape == gt.shape, f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"

    if pred.ndim == 2:
        pred_slice = pred
        gt_slice = gt
    elif pred.ndim == 3:
        if slice_idx is None:
            sums = [np.sum((gt[:, :, i] == 1) | (gt[:, :, i] == 2)) for i in range(gt.shape[2])]
            slice_idx = np.argmax(sums)
        pred_slice = pred[:, :, slice_idx]
        gt_slice = gt[:, :, slice_idx]
    else:
        raise ValueError("Unsupported shape")

    overlay = np.zeros((*gt_slice.shape, 3), dtype=np.uint8)

    # Best Plane (label=1)
    tp1 = (gt_slice == 1) & (pred_slice == 1)
    fn1 = (gt_slice == 1) & (pred_slice != 1)
    fp1 = (gt_slice != 1) & (pred_slice == 1)

    # Sub-Best Plane (label=2)
    tp2 = (gt_slice == 2) & (pred_slice == 2)
    fn2 = (gt_slice == 2) & (pred_slice != 2)
    fp2 = (gt_slice != 2) & (pred_slice == 2)

    # overlay[tp1] = [0, 255, 0]      # Green
    # overlay[fn1] = [255, 0, 0]      # Red
    # overlay[fp1] = [0, 0, 255]      # Blue
    # overlay[tp2] = [255, 255, 0]    # Yellow
    # overlay[fn2] = [255, 165, 0]    # Orange
    # overlay[fp2] = [128, 0, 128]    # Purple

    overlay[tp1] = [255, 255, 0]      # ✅ Green → 现在是 label=2 正确预测
    overlay[fn1] = [255, 0, 0]      # ✅ Red → label=2 漏检
    overlay[fp1] = [0, 0, 255]      # ✅ Blue → label=2 错报

    overlay[tp2] =[0, 255, 0]    # Yellow → label=1 正确预测
    overlay[fn2] = [255, 165, 0]    # Orange → label=1 漏检
    overlay[fp2] = [128, 0, 128]    # Purple → label=1 错报

    # 显示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_slice, cmap='gray')
    axes[0].set_title("Ground Truth")

    axes[1].imshow(pred_slice, cmap='gray')
    axes[1].set_title("Prediction")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis('off')

    # 添加图例

    # legend_patches = [
    #     mpatches.Patch(color=[0/255,255/255,0/255],   label="TP1 - Best Plane Correct"),
    #     mpatches.Patch(color=[1,0,0],                 label="FN1 - Best Plane Missed"),
    #     mpatches.Patch(color=[0,0,1],                 label="FP1 - Best Plane False"),
    #     mpatches.Patch(color=[1,1,0],                 label="TP2 - Sub-Best Correct"),
    #     mpatches.Patch(color=[1,165/255,0],           label="FN2 - Sub-Best Missed"),
    #     mpatches.Patch(color=[128/255, 0, 128/255],   label="FP2 - Sub-Best False"),
    # ]
    legend_patches = [
        mpatches.Patch(color=[0/255,255/255,0/255],   label="TP1 - Best Plane Correct"),
        mpatches.Patch(color=[1,0,0],                 label="FN1 - Sub-Best Missed"),
        mpatches.Patch(color=[0,0,1],                 label="FP1 - Sub-Best False"),
        mpatches.Patch(color=[1,1,0],                 label="TP2 - Sub-Best Correct"),
        mpatches.Patch(color=[1,165/255,0],           label="FN2 - Best Plane Missed"),
        mpatches.Patch(color=[128/255, 0, 128/255],   label="FP2 - Best Plane False"),
    ]

    axes[2].legend(handles=legend_patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

# ---------- 批量可视化 ----------
for i, case in enumerate(sorted_cases):
    pred_file = case['prediction_file']
    gt_file = case['reference_file']
    filename = os.path.basename(pred_file).replace('.nii.gz', f'_vis.png')
    save_path = os.path.join(SAVE_DIR, f"top{i+1}_{filename}")
    print(f"Visualizing case {i+1}: {filename}")
    visualize_case_multilabel(pred_file, gt_file, save_path)