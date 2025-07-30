"""

Created on 2025/7/26 03:39
@author: 18310

"""
import os
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

# 配置路径（改成你的实际路径）
images_dir = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset047_ACOptimalSuboptimal_slice2000_2d\imagesTr")
labels_dir = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset047_ACOptimalSuboptimal_slice2000_2d\labelsTr")

# 列出所有文件名（不包含后缀）
# 列出所有文件名（提取纯 ID，如 'xxx_0032'）
all_ids = [f.name.replace("_0000.nii.gz", "") for f in images_dir.glob("*_0000.nii.gz")]

# 识别正负样本（依据标签中是否含正类）
pos_ids, neg_ids = [], []
for id_ in all_ids:
    label_path = labels_dir / f"{id_}.nii.gz"
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))
    if np.any(label > 0):
        pos_ids.append(id_)
    else:
        neg_ids.append(id_)

# 随机抽取若干个正负样本
N = 3
sample_pos = random.sample(pos_ids, min(N, len(pos_ids)))
sample_neg = random.sample(neg_ids, min(N, len(neg_ids)))

# 显示函数：图像 / mask / overlay
def visualize_sample(image_path, label_path, title):
    import matplotlib.pyplot as plt
    import numpy as np
    import SimpleITK as sitk
    from matplotlib import colors

    img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))

    img = np.squeeze(img)
    lbl = np.squeeze(lbl)

    # 设置掩码颜色：0=黑色，1=鲜绿
    green_cmap = colors.ListedColormap(['black', '#00FF00'])
    masked = np.ma.masked_where(lbl == 0, lbl)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title} - Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(masked, cmap=green_cmap, vmin=0, vmax=1)
    plt.title(f"{title} - Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap='gray')
    plt.imshow(masked, cmap=green_cmap, vmin=0, vmax=1, alpha=0.5)
    plt.title(f"{title} - Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# 可视化正样本
for sid in sample_pos:
    visualize_sample(images_dir / f"{sid}_0000.nii.gz", labels_dir / f"{sid}.nii.gz", title=f"Positive: {sid}")

# 可视化负样本
for sid in sample_neg:
    visualize_sample(images_dir / f"{sid}_0000.nii.gz", labels_dir / f"{sid}.nii.gz", title=f"Negative: {sid}")
