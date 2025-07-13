import os
import csv
from pathlib import Path
import SimpleITK as sitk
import numpy as np

def count_slices_in_case(img_path: Path, mask_path: Path):
    """
    统计单个 case 中的各类帧数量，返回 (total, pure_black, annotated, unannotated)
    """
    # 读入整体 3D volume，shape = (slices, H, W)
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
    if mask_path.exists():
        msk = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    else:
        # 如果没有 mask，就用全 0 填充
        msk = np.zeros_like(img, dtype=np.uint8)

    total = img.shape[0]

    # 按帧计算
    # 1. 纯黑帧：该 slice 所有像素都 == 0
    pure_black = int(np.all(img.reshape(total, -1) == 0, axis=1).sum())
    # 2. 有标注帧：图像不全黑且 mask 中有任意 >0
    non_black = ~np.all(img.reshape(total, -1) == 0, axis=1)
    has_mask  = np.any(msk.reshape(total, -1) > 0, axis=1)
    annotated = int((non_black & has_mask).sum())
    # 3. 未标注帧：图像非纯黑但 mask 全零
    unannotated = int((non_black & ~has_mask).sum())

    return total, pure_black, annotated, unannotated

def fmt_pct(count: int, total: int) -> str:
    """格式化百分比字符串"""
    return f"{count / total * 100:5.2f}%"

if __name__ == "__main__":
    IMAGES_DIR = Path(r"E:\Data Science Master Project-code\ACOUSLIC-AI-baseline\test\input\images\stacked-fetal-ultrasound")
    MASKS_DIR  = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
    OUTPUT_CSV = Path("slice_statistics.csv")

    # 准备写 CSV
    fieldnames = [
        "case_id",
        "total_frames",
        "pure_black",
        "pct_black",
        "annotated",
        "pct_annotated",
        "unannotated",
        "pct_unannotated"
    ]
    records = []

    overall = {"total":0, "black":0, "pos":0, "neg":0}

    # 遍历每个 mha 文件
    for img_path in sorted(IMAGES_DIR.glob("*.mha")):
        case_id   = img_path.stem
        mask_path = MASKS_DIR / f"{case_id}.mha"

        if not mask_path.exists():
            print(f"[warn] Case {case_id} 未找到标签，所有非纯黑帧计为“未标注”")

        total, black, pos, neg = count_slices_in_case(img_path, mask_path)

        overall["total"] += total
        overall["black"] += black
        overall["pos"]   += pos
        overall["neg"]   += neg

        records.append({
            "case_id":       case_id,
            "total_frames":  total,
            "pure_black":    black,
            "pct_black":     fmt_pct(black, total),
            "annotated":     pos,
            "pct_annotated": fmt_pct(pos, total),
            "unannotated":   neg,
            "pct_unannotated": fmt_pct(neg, total)
        })

        pct_black     = fmt_pct(black, total)
        pct_annotated = fmt_pct(pos, total)
        pct_unannotated = fmt_pct(neg, total)

        # **打印每个 case 的统计结果**
        print(f"Case {case_id}: total={total}, pure_black={black} ({pct_black}), "
              f"annotated={pos} ({pct_annotated}), unannotated={neg} ({pct_unannotated})")
    # 添加全局统计行
    T = overall["total"]
    B = overall["black"]
    P = overall["pos"]
    N = overall["neg"]
    print(f"总帧数            : {T}")
    print(f"纯黑帧（全零）    : {B} ({fmt_pct(B, T)})")
    print(f"有标注帧（正样本）: {P} ({fmt_pct(P, T)})")
    print(f"未标注帧（负样本）: {N} ({fmt_pct(N, T)})")
    records.append({
        "case_id":         "ALL",
        "total_frames":    T,
        "pure_black":      B,
        "pct_black":       fmt_pct(B, T),
        "annotated":       P,
        "pct_annotated":   fmt_pct(P, T),
        "unannotated":     N,
        "pct_unannotated": fmt_pct(N, T)
    })

    # 写 CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"统计结果已保存到 {OUTPUT_CSV.resolve()}")

    # 简单校验
    assert T == B + P + N, "统计不一致：总帧数 != 纯黑 + 有标注 + 未标注"