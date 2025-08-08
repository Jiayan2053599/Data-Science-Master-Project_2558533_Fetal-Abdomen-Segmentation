"""

Created on 2025/7/27 03:36
@author: 18310

"""
import os
import subprocess

# 设置 nnUNet 路径
os.environ["nnUNet_raw"] = r"D:\nnUNet\nnUNet_raw_data_base"
os.environ["nnUNet_preprocessed"] = r"D:\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"D:\nnUNet\nnUNet_results"

# CPU 多线程优化（适配 Ryzen 9 + 32GB）
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["nnUNet_n_proc_DA"] = "4"  # 训练时数据增强用

# 构造命令：预处理（使用 2 进程）
cmd_preprocess = (
    "nnUNetv2_plan_and_preprocess "
    "-d 303 "
    "-np 2 "
    "--verify_dataset_integrity "
    "--verbose"
)
cmd_train = (

    "nnUNetv2_train "
    "303 3d_fullres 0 "
    "-tr MyTrainer "
    "--npz --use_compressed -num_gpus 1"
)

# 定义函数：实时执行并输出每行日志
def run_command(cmd, title):
    print(f"\n{'='*10} {title} {'='*10}\n")
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          bufsize=1, universal_newlines=True, encoding="utf-8") as p:
        for line in p.stdout:
            print(line, end='')

# 执行预处理流程
run_command(cmd_preprocess, "Preprocessing")

# 执行训练流程
run_command(cmd_train, "Training")
