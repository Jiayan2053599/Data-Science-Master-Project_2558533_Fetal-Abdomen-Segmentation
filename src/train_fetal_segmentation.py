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

# ─── 在脚本里硬编码 nnU-Net 要求的环境变量 ───────────────────────
# Windows 示例：
os.environ["nnUNet_raw_data_base"]          = r"D:\nnUNet\nnUNet_raw_data_base"
os.environ["nnUNet_preprocessed"]          = r"D:\nnUNet\nnUNet_preprocessed"
os.environ["nnUNet_results"]               = r"D:\nnUNet\results"

# Linux / EC2 示例：
# os.environ["nnUNet_raw_data_base"]          = "/home/ec2-user/nnUNet/nnUNet_raw_data_base"
# os.environ["nnUNet_preprocessed"]          = "/home/ec2-user/nnUNet/nnUNet_preprocessed"
# os.environ["nnUNet_results"]               = "/home/ec2-user/nnUNet/nnUNet_results"
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

if __name__ == "__main__":
    # 打印环境变量，便于调试
    print(">> nnUNet_raw_data_base =", os.environ.get("nnUNet_raw_data_base"))
    print(">> nnUNet_preprocessed   =", os.environ.get("nnUNet_preprocessed"))
    print(">> nnUNet_results        =", os.environ.get("nnUNet_results"))
    print(">> CUDA_VISIBLE_DEVICES  =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(">> nnUNet_n_proc_DA      =", os.environ.get("nnUNet_n_proc_DA"))
    print(">> OMP_NUM_THREADS       =", os.environ.get("OMP_NUM_THREADS"))
    print(">> MKL_NUM_THREADS       =", os.environ.get("MKL_NUM_THREADS"))
    print()

    # 构造训练命令
    cmd = [
        "nnUNetv2_train",
        # "300",               # 数据集 ID
        "Dataset300_ACOptimalSuboptimal_slice2000",
        "2d",                # 只跑 2d
        "0",                 # fold 0
        "-tr", "MyTrainer",# 自定义 Trainer
        "--npz",             # 保存 softmax 概率
        "--use_compressed",  # 直接读 .npz
        "-num_gpus", "1",   # 使用一张 GPU
    ]
    print(">>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ)