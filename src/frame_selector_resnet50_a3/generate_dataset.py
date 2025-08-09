import os
import numpy as np
import SimpleITK as sitk
import csv
from pathlib import Path
from tqdm import tqdm

# ====================== 配置路径 ======================
mha_img_dir = Path(r"D:/Data_Science_project-data/acouslic-ai-train-set/images/stacked-fetal-ultrasound")
mha_label_dir = Path(r"D:/Data_Science_project-data/acouslic-ai-train-set/masks/stacked_fetal_abdomen")
output_img_dir = Path(r"D:/Resnet/imagesTr")
output_lbl_dir = Path(r"D:/Resnet/labelsTr")
csv_path = Path(r"E:/Data Science Master Project-code/ACOUSLIC-AI-baseline/src/frame_selector_resnet50_a3/frame_labels.csv")

output_img_dir.mkdir(parents=True, exist_ok=True)
output_lbl_dir.mkdir(parents=True, exist_ok=True)

MAX_FILES = 100  # 控制处理的 .mha 文件数量

# ====================== 主流程 ======================
all_mha_files = sorted(mha_img_dir.glob("*.mha"))[:MAX_FILES]
frame_records = []
total_abdomen, total_frames = 0, 0
skip_cases = 0

for mha_path in tqdm(all_mha_files, desc="处理样本"):
    case_id = mha_path.stem
    label_path = mha_label_dir / f"{case_id}.mha"

    if not label_path.exists():
        print(f"[跳过] 缺失标签：{label_path}")
        skip_cases += 1
        continue

    try:
        img_3d = sitk.ReadImage(str(mha_path))
        lbl_3d = sitk.ReadImage(str(label_path))
        img_np = sitk.GetArrayFromImage(img_3d)
        lbl_np = sitk.GetArrayFromImage(lbl_3d).astype(np.uint8)
    except Exception as e:
        print(f"[跳过] {case_id} 读取失败: {e}")
        skip_cases += 1
        continue

    spacing = img_3d.GetSpacing()[:2]
    origin = img_3d.GetOrigin()[:2]
    direction_full = img_3d.GetDirection()
    if len(direction_full) == 9:
        direction_2d = (direction_full[0], direction_full[1], direction_full[3], direction_full[4])
    else:
        direction_2d = (1.0, 0.0, 0.0, 1.0)

    for i in range(img_np.shape[0]):
        img_slice_np = img_np[i]
        lbl_slice_np = lbl_np[i]

        is_abdomen = int(np.any(np.isin(lbl_slice_np, [1, 2])))
        total_abdomen += is_abdomen
        total_frames += 1

        img_slice_itk = sitk.GetImageFromArray(img_slice_np)
        lbl_slice_itk = sitk.GetImageFromArray(lbl_slice_np)

        img_slice_itk.SetSpacing(spacing)
        img_slice_itk.SetOrigin(origin)
        img_slice_itk.SetDirection(direction_2d)

        lbl_slice_itk.SetSpacing(spacing)
        lbl_slice_itk.SetOrigin(origin)
        lbl_slice_itk.SetDirection(direction_2d)

        out_img_name = f"{case_id}_{i:04d}_0000.nii.gz"
        out_lbl_name = f"{case_id}_{i:04d}.nii.gz"

        sitk.WriteImage(img_slice_itk, str(output_img_dir / out_img_name))
        sitk.WriteImage(lbl_slice_itk, str(output_lbl_dir / out_lbl_name))

        frame_records.append([out_img_name, is_abdomen])

# ====================== 写入 CSV 标签 ======================
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(frame_records)

# ====================== 汇总输出 ======================
print("\n=== 完成 ===")
print(f"总帧数：{total_frames}")
print(f"含标注帧数（label=1）：{total_abdomen}")
print(f"跳过样本数（无标签或失败）：{skip_cases}")
print(f"图像输出目录：{output_img_dir}")
print(f"标签输出目录：{output_lbl_dir}")
print(f"CSV 路径：{csv_path}")
