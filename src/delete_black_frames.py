"""

Created on 2025/7/9 00:31
@author: 18310

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_black_slices.py

遍历一批 .mha 体数据，去掉全黑帧，并保存新的 image 与 mask。
"""

import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np

def process_case(img_path: Path, mask_path: Path, out_img_path: Path, out_mask_path: Path):
    """
    读取 img_path，对全零 slice 进行过滤，并写出到 out_img_path。
    如果 mask_path 存在，则同样过滤并写出到 out_mask_path。
    """
    # 读取 image
    img = sitk.ReadImage(str(img_path))
    arr = sitk.GetArrayFromImage(img)  # numpy array with shape (D, H, W)

    # 找到哪些 slice 不是全零
    non_black = np.any(arr != 0, axis=(1, 2))  # bool array, length D

    # 如果所有 slice 都全零，跳过
    if not np.any(non_black):
        print(f"[skip-all-black] {img_path.name} 全部切片均为 0，跳过")
    else:
        # 过滤并写出 image
        arr_filt = arr[non_black]
        img_filt = sitk.GetImageFromArray(arr_filt)
        img_filt.SetSpacing(img.GetSpacing())
        img_filt.SetOrigin(img.GetOrigin())
        img_filt.SetDirection(img.GetDirection())
        sitk.WriteImage(img_filt, str(out_img_path), useCompression=True)
        print(f"[saved-img] {out_img_path}")

    # 处理 mask（如果存在）
    if mask_path.exists():
        mask = sitk.ReadImage(str(mask_path))
        m_arr = sitk.GetArrayFromImage(mask)
        # 使用同样的 non_black 索引过滤 mask
        m_filt = m_arr[non_black]
        mask_filt = sitk.GetImageFromArray(m_filt)
        mask_filt.SetSpacing(mask.GetSpacing())
        mask_filt.SetOrigin(mask.GetOrigin())
        mask_filt.SetDirection(mask.GetDirection())
        sitk.WriteImage(mask_filt, str(out_mask_path), useCompression=True)
        print(f"[saved-msk] {out_mask_path}")
    else:
        print(f"[warn] 未找到 mask：{mask_path.name}，只处理 image")

def main():
    # TODO: 根据实际路径修改下面三个目录
    IMAGES_DIR      = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal\imagesTr")
    MASKS_DIR       = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset300_ACOptimalSuboptimal\labelsTr")
    OUT_IMAGES_DIR  = Path(r"D:\processed\imagesTr_nonblack")
    OUT_MASKS_DIR   = Path(r"D:\processed\labelsTr_nonblack")

    # 创建输出目录
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # 遍历所有 .mha
    img_files = sorted(IMAGES_DIR.glob("*.mha"))
    print(f"Found {len(img_files)} image files. Processing...")

    for idx, img_path in enumerate(img_files, 1):
        case_id = img_path.stem  # e.g. 'volume000'
        mask_path = MASKS_DIR / f"{case_id}.mha"
        out_img_path  = OUT_IMAGES_DIR / img_path.name
        out_mask_path = OUT_MASKS_DIR  / f"{case_id}.mha"

        print(f"[{idx}/{len(img_files)}] Processing case {case_id}...")
        process_case(img_path, mask_path, out_img_path, out_mask_path)

    print("All done.")

if __name__ == "__main__":
    main()
