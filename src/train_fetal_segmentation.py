# import sys
# import importlib
#
# # ==============================================
# # Monkey‐patch numpy._core 让 pickle 找得到
# # ==============================================
# # 把 numpy.core 模块放到 sys.modules['numpy._core']
# np_core = importlib.import_module("numpy.core")
# sys.modules["numpy._core"] = np_core
#
# # alias 多个子模块，否则可能还报 multiarray/_multiarray_umath 找不到
# sys.modules["numpy._core.multiarray"]        = importlib.import_module("numpy.core.multiarray")
# sys.modules["numpy._core._multiarray_umath"] = importlib.import_module("numpy.core._multiarray_umath")
# # 如果 pickle 里还要别的子模块，按同样方式 alias
# # sys.modules["numpy._core._multiarray_tests"] = importlib.import_module("numpy.core._multiarray_tests")
# # ==============================================
import os
import subprocess
import multiprocessing

# ─── 在脚本里硬编码 nnU-Net 要求的环境变量 ────
# Windows 示例：
os.environ["nnUNet_raw_data_base"]         = r"D:\nnUNet\nnUNet_raw_data_base"
os.environ["nnUNet_preprocessed"]          = r"D:\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"]               = r"D:\nnUNet\results"

# Linux / EC2 示例：
# os.environ["nnUNet_raw_data_base"]          = "/home/ec2-user/nnUNet/nnUNet_raw_data_base"
# os.environ["nnUNet_preprocessed"]           = "/home/ec2-user/nnUNet/nnUNet_preprocessed"
# os.environ["nnUNet_results"]                = "/home/ec2-user/nnUNet/nnUNet_results"
# ────────────────────────────────────────────────────────────────────

# ─── CUDA & 并行设置 ────────────────────────────────────────────────
# 仅使用 GPU:0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # DA 进程数：不超过总核数的一半
# total_cpus = multiprocessing.cpu_count()
# n_da = max(1, total_cpus // 2)
# os.environ["nnUNet_n_proc_DA"]     = str(n_da)
# 禁用多进程
os.environ["nnUNet_n_proc_DA"]     = "0"
# 限制每个进程线程数，防止过多 context switch
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
# ────────────────────────────────────────────────────────────────────

TASK_ID = "300"  # 你的任务 ID，数字
FOLD = "0"       # 通常训练 fold=0
CONFIG = "2d"    # 配置为 2D
TRAINER = "MyTrainer"  # 自定义 Trainer 名字

if __name__ == "__main__":
    print("== 环境变量配置 ==")
    print("nnUNet_raw =", os.environ["nnUNet_raw"])
    print("nnUNet_preprocessed =", os.environ["nnUNet_preprocessed"])
    print("nnUNet_results =", os.environ["nnUNet_results"])
    print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
    print()

    # Step 1: 执行 2D 配置的 fingerprint + planning + preprocessing
    preprocess_cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", TASK_ID,
        "--verify_dataset_integrity",
        "-np", "4",      # fingerprint 并行数
        "-npfp", "4",    # preprocessing 并行数
        "--verbose"
    ]
    print(">>> Running preprocessing:", " ".join(preprocess_cmd))
    subprocess.run(preprocess_cmd, check=True, env=os.environ)

    # Step 2: 开始训练流程
    train_cmd = [
        "nnUNetv2_train",
        CONFIG,              # 2d 配置
        TASK_ID,             # 任务编号
        FOLD,                # 训练 fold
        "-tr", TRAINER,      # Trainer 类名
        "--npz",             # 保存 softmax 输出（用于 ensemble）
        "--use_compressed",  # 直接读取压缩的 .npz
        "-num_gpus", "1"     # 使用一张 GPU
    ]
    print(">>> Running training:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True, env=os.environ)