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


# ─── 在脚本里硬编码好 nnU-Net 要求的环境变量 ───────────────────────
# os.environ["nnUNet_raw"]          = r"D:\nnUNet\nnUNet_raw_data_base"
# os.environ["nnUNet_preprocessed"] = r"D:\nnUNet\nnUNet_preprocessed"
# os.environ["nnUNet_results"]      = r"D:\nnUNet\results"

os.environ["nnUNet_raw"]          = "/home/ec2-user/nnUNet/nnUNet_raw_data_base"
os.environ["nnUNet_preprocessed"] = "/home/ec2-user/nnUNet/nnUNet_preprocessed"
os.environ["nnUNet_results"]      = "/home/ec2-user/nnUNet/nnUNet_results"
# ────────────────────────────────────────────────────────────────

# ─── 硬编码 CPU/GPU 并行设置 ────────────────────────
# 只让脚本在第 0 号 GPU 上跑
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# nnU-Net 采样增强进程数：不超过总核数的一半
total_cpus = multiprocessing.cpu_count()  # 例如 8
n_da = max(1, total_cpus // 2)
os.environ["nnUNet_n_proc_DA"] = str(n_da)

# 限制每个进程的 OpenMP/ MKL 线程数，避免过多上下文切换
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ────────────────────────────────────────────────────

if __name__ == "__main__":
    # 构造调用命令，完全等价于你在终端里敲：
    # nnUNetv2_train 300 2d 0 --npz --use_compressed --num_gpus 1 --max_epochs 200 --val_every_n_epochs 10

    # cmd = [
    #     "nnUNetv2_train",
    #     "300",               # 数据集 ID
    #     "2d",                # 只跑 2d
    #     "0",                 # fold 0
    #     "-tr", "MyTrainer",  # 指定自定义的子类
    #     "--npz",             # 保存 softmax 概率
    #     "--use_compressed",  # 读 .npz，不解包 .npy
    #     "-num_gpus", "1"     # 使用一张 GPU
    # ]

    # sanity check
    print(">> nnUNet_raw          =", os.environ["nnUNet_raw"])
    print(">> nnUNet_preprocessed =", os.environ["nnUNet_preprocessed"])
    print(">> nnUNet_results      =", os.environ["nnUNet_results"])
    print(">> CUDA_VISIBLE_DEVICES=", os.environ["CUDA_VISIBLE_DEVICES"])
    print(">> nnUNet_n_proc_DA    =", os.environ["nnUNet_n_proc_DA"])
    print(">> OMP_NUM_THREADS     =", os.environ["OMP_NUM_THREADS"])
    print(">> MKL_NUM_THREADS     =", os.environ["MKL_NUM_THREADS"])
    print()

    cmd = [
        "nnUNetv2_train",
        "300",  # 数据集 ID
        "2d",  # 只跑 2d
        "0",  # fold 0
        "-tr", "MyTrainer",  # 指定自定义 Trainer
        "--npz",  # 保存 softmax 概率
        "--use_compressed",  # 直接读 .npz
        "-num_gpus", "1"  # 使用一张 GPU
    ]
    print(">>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ)
