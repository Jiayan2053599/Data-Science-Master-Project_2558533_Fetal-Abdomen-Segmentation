"""

Created on 2025/7/29 04:44
@author: 18310

"""

"""
generate_dataset.py

功能说明：
- 遍历 nnUNet 的 imagesTr 和 labelsTr（.nii.gz 格式）
- 每一帧图像是否含有 abdomen（标签中有 1 或 2） → 二分类标签（0/1）
- 保存帧为 PNG 图像，并输出 frame_labels.csv

依赖库：
pip install nibabel opencv-python tqdm
"""

import os
import nibabel as nib
import numpy as np
import cv2
import csv
from tqdm import tqdm

# ====================== 配置部分 ======================
imagesTr_dir = r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal\imagesTr"
labelsTr_dir = r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal\labelsTr"
output_img_dir = r"E:\Data Science Master Project-code\ACOUSLIC-AI-baseline\frame_images"
csv_path = r"E:\Data Science Master Project-code\ACOUSLIC-AI-baseline\src\frame_selector_resnet50_a3\frame_labels.csv"

os.makedirs(output_img_dir, exist_ok=True)

# ====================== 主处理流程 ======================
frame_records = []
total_abdomen, total_frames = 0, 0
skip_cases = 0

for fname in tqdm(os.listdir(imagesTr_dir), desc="正在处理样本"):
    if not fname.endswith(".nii.gz") or "_0000" not in fname:
        continue

    case_id = fname.replace("_0000.nii.gz", "")
    img_path = os.path.join(imagesTr_dir, fname)
    lbl_path = os.path.join(labelsTr_dir, f"{case_id}.nii.gz")

    if not os.path.exists(lbl_path):
        print(f"缺失标签：{lbl_path}")
        skip_cases += 1
        continue

    try:
        img = np.asarray(nib.load(img_path).dataobj)  # shape: (H, W, D)
        lbl = np.asarray(nib.load(lbl_path).dataobj).astype(np.uint8)  # 转为整数
    except Exception as e:
        print(f"读取失败：{case_id}, 错误：{e}")
        skip_cases += 1
        continue

    D = img.shape[2]
    for i in range(D):
        img_slice = img[:, :, i]
        lbl_slice = lbl[:, :, i]

        # 腹部标注判断（仅考虑标签 1 和 2）
        is_abdomen = int(np.any((lbl_slice == 1) | (lbl_slice == 2)))
        total_abdomen += is_abdomen
        total_frames += 1

        # 图像归一化为 0–255
        min_val, max_val = np.min(img_slice), np.max(img_slice)
        if max_val - min_val < 1e-5:
            img_norm = np.zeros_like(img_slice, dtype=np.uint8)
        else:
            img_norm = ((img_slice - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)

        filename = f"{case_id}_{i:04d}.png"
        cv2.imwrite(os.path.join(output_img_dir, filename), img_norm)
        frame_records.append([filename, is_abdomen])

# 写入标签 CSV 文件
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(frame_records)

# ====================== 完成信息 ======================
print("\n帧图像提取完成")
print(f"图像保存目录：{output_img_dir}")
print(f"标签 CSV 文件：{csv_path}")
print(f"总帧数：{total_frames}，其中腹部帧（label=1）数量：{total_abdomen}")
print(f"跳过样本数量（无标签或错误）：{skip_cases}")


# import nibabel as nib
# import numpy as np
#
# label_path = r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal\labelsTr\0d0a3298-a9c6-43c3-a9e3-df3a9c0afa06.nii.gz"
# label_img = nib.load(label_path)
# label_data = label_img.get_fdata()
#
# # 强制转为整数
# label_int = label_data.astype(np.uint8)
# unique_vals = np.unique(label_int)
#
# print("标签值种类（整数转换后）:", unique_vals)