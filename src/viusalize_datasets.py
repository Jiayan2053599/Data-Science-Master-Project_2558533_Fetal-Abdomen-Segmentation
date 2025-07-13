"""

Created on 2025/7/3 02:47
@author: 18310

"""
import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# ─── 用户配置 ──────────────────────────────────────────────────────────
RAW_BASE           = Path(r"D:\nnUNet\nnUNet_raw_data_base")  # raw_data_base 根目录
TASK_ID            = 300
TASK_NAME          = "ACOptimalSuboptimal"
NUM_CASES          = 2   # 前 N 例要可视化
NUM_FRAMES_TO_SHOW = 5   # 每例只展示有 ROI 的前 N 帧
# ────────────────────────────────────────────────────────────────────────

def load_nifti(path: Path) -> np.ndarray:
    """读 .nii/.nii.gz → numpy array"""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr

def visualize_roi_frames(img_arr: np.ndarray, lbl_arr: np.ndarray, max_frames=5):
    """
    只可视化那些 mask.max()>0 的帧，最多 max_frames 帧。
    img_arr, lbl_arr: numpy.ndarray, shape (T,H,W)
    """
    # 找到含 ROI 的帧索引
    per_frame_max = lbl_arr.reshape(lbl_arr.shape[0], -1).max(axis=1)
    roi_frames = np.where(per_frame_max > 0)[0]
    if roi_frames.size == 0:
        print("  该序列中没有含 ROI 的帧，跳过。")
        return

    print(f"  Found {len(roi_frames)} ROI frames; showing first {min(len(roi_frames), max_frames)}:")
    for t in roi_frames[:max_frames]:
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        fig.suptitle(f"Frame {t}", fontsize=14)

        # 灰度图
        axes[0].imshow(img_arr[t], cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        # ROI overlay
        mask = lbl_arr[t]
        masked = np.ma.masked_where(mask == 0, mask.astype(float) / mask.max())
        axes[1].imshow(img_arr[t], cmap="gray", alpha=1.0)
        axes[1].imshow(masked, cmap="Accent", alpha=0.7)
        axes[1].set_title("Overlay Label")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 1) 这里改回 TaskXXX_YYY，与 convert.py/prepare 脚本保持一致
    task_folder = RAW_BASE / f"Dataset{TASK_ID:03d}_{TASK_NAME}"
    imagesTr    = task_folder / "imagesTr"
    labelsTr    = task_folder / "labelsTr"

    # 2) 检查目录是否存在
    if not imagesTr.is_dir() or not labelsTr.is_dir():
        raise RuntimeError(f"请先确认目录存在：\n  {imagesTr}\n  {labelsTr}")

    # 3) 列出所有 *_0000.nii*，取前 NUM_CASES 个
    img_files = sorted(imagesTr.glob("*_0000.nii*"))[:NUM_CASES]

    for img_path in img_files:
        base = img_path.stem.split("_0000")[0]
        # 尝试标签文件 .nii.gz 或 .nii
        lbl_path = None
        for ext in (".nii.gz", ".nii"):
            cand = labelsTr / f"{base}{ext}"
            if cand.exists():
                lbl_path = cand
                break
        if lbl_path is None:
            print(f"[warn] 找不到标签文件 for {base}")
            continue

        # 4) 读取 numpy
        img_arr = load_nifti(img_path)
        lbl_arr = load_nifti(lbl_path)
        print(f"Case {base} — image shape: {img_arr.shape}, label shape: {lbl_arr.shape}")

        # 5) 只对 cine 序列 (3D) 做 ROI 可视化
        if img_arr.ndim == 3 and lbl_arr.ndim == 3:
            visualize_roi_frames(img_arr, lbl_arr, max_frames=NUM_FRAMES_TO_SHOW)
        else:
            print(f"[info] 跳过非 3D cine 数据 (ndim={img_arr.ndim}) for {base}")