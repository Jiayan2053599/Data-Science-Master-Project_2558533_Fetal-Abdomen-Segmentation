import os
import subprocess

# 设置环境变量
os.environ["nnUNet_raw"] = "D:/nnUNet/nnUNet_raw_data_base"
os.environ["nnUNet_preprocessed"] = "D:/nnUNet/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "D:/nnUNet/nnUNet_results"

# 数据集编号与配置
dataset_id = "047"   # "047"
config = "2d"
fold = "0"

print("Step 1: Reading dataset.json from:", os.path.join(os.environ["nnUNet_raw"], f"Dataset{dataset_id}_AC_mha2d_top2000", "dataset.json"))

# Step 2: 预处理命令
preprocess_cmd = [
    "nnUNetv2_plan_and_preprocess",
    "-d", dataset_id,
    "-c", config,
    "--verify_dataset_integrity",
    "--clean",
    "--verbose"
]

print("\nStep 2: Running preprocessing command:")
print(">", " ".join(preprocess_cmd))
preprocess_result = subprocess.run(preprocess_cmd, capture_output=True, text=True)

# 输出日志
print(preprocess_result.stdout)
print(preprocess_result.stderr)

# Step 3: 成功后执行训练
if preprocess_result.returncode == 0:
    print("Preprocessing complete. Starting training...\n")

    train_cmd = [
        "nnUNetv2_train",
        dataset_id,
        config,
        fold,
        "-tr", "MyTrainer",
        "--npz",
        "--use_compressed",
        "-num_gpus", "1"
    ]

    log_file = f"train_log_fold{fold}.txt"
    with open(log_file, "w", encoding="utf-8") as logf:
        process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")
            logf.write(line)

    process.wait()
    if process.returncode == 0:
        print(f"\nTraining finished. Log saved to: {log_file}")
    else:
        print(f"\nTraining failed. Please check log: {log_file}")
else:
    print("Preprocessing failed. Please check dataset.json and file paths.")


# import json
# from pathlib import Path
#
# dataset_json = {
#     "name": "Dataset047_ACOptimalSuboptimal_slice2000",
#     "description": "2D slice subset with optimal/suboptimal labels",
#     "tensorImageSize": "2D",
#     "reference": "",
#     "licence": "",
#     "release": "1.0",
#     "modality": {
#         0: "ULTRASOUND"
#     },
#     "channel_names": {
#         0: "ultrasound"
#     },
#     "labels": {
#         0: "background",
#         1: "optimal_surface",
#         2: "suboptimal_surface"
#     },
#     "file_ending": ".nii.gz",
#     "numTraining": 2000,
#     "numTest": 0
# }
#
# # 保存路径
# output_path = Path(r"D:\nnUNet\nnUNet_raw_data_base\Dataset047_ACOptimalSuboptimal_slice2000_2d\dataset.json")
#
# # 确保使用 ensure_ascii=False 以保留 UTF-8，indent 方便调试
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(dataset_json, f, indent=2, ensure_ascii=False)
