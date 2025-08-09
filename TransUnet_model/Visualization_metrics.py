from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

# 根日志目录
root_log_dir = r"D:\TransUnet\trainingrecords_transUnet\log\mydata_transunet_Lovasz"

# key: 图例名称; value: (子目录名, tag 名)
metrics_to_plot = {
    # ——————— 训练集 ———————
    "Train Accuracy":    ("train_metrics_ac",    "train/metrics"),
    "Train Mean DSC":    ("train_metrics_mdsc",  "train/metrics"),
    "Train F1":          ("train_metrics_mf1",   "train/metrics"),
    "Train mIoU":        ("train_metrics_miou",  "train/metrics"),
    "Train Precision":   ("train_metrics_mpc",   "train/metrics"),
    "Train MSE":         ("train_metrics_mse",   "train/metrics"),
    "Train MSP":         ("train_metrics_msp",   "train/metrics"),

    # ——————— 验证集整体 ———————
    "Val Accuracy":      ("val_metrics_ac",      "val/metrics"),
    "Val Mean DSC":      ("val_metrics_mdsc",    "val/metrics"),
    "Val F1":            ("val_metrics_mf1",     "val/metrics"),
    "Val mIoU":          ("val_metrics_miou",    "val/metrics"),
    "Val Precision":     ("val_metrics_mpc",     "val/metrics"),
    "Val MSE":           ("val_metrics_mse",     "val/metrics"),
    "Val MSP":           ("val_metrics_msp",     "val/metrics"),

    # ——————— 验证集分类别 ———————
    "Val Class0 IoU":    ("val_class_iou_class0", "val/class_iou"),
    "Val Class1 IoU":    ("val_class_iou_class1", "val/class_iou"),
    "Val Class2 IoU":    ("val_class_iou_class2", "val/class_iou"),
    "Val mIoU (all)":    ("val_class_iou_miou",   "val/class_iou"),
}

all_series = {}
for legend, (subdir, tag) in metrics_to_plot.items():
    log_dir = os.path.join(root_log_dir, subdir)
    # 找到唯一的 events 文件
    files = [f for f in os.listdir(log_dir) if f.startswith("events.out")]
    if not files:
        print(f"{log_dir} 下没找到 event 文件，跳过 {legend}")
        continue
    ev_path = os.path.join(log_dir, files[0])

    ea = event_accumulator.EventAccumulator(ev_path,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"{ev_path} 中没找到 tag `{tag}`，可用 tags = {ea.Tags()['scalars']}")
        continue

    events = ea.Scalars(tag)
    steps = [e.step  for e in events]
    vals  = [e.value for e in events]
    all_series[legend] = (steps, vals)

# 绘图
plt.figure(figsize=(10,6))
for name, (xs, ys) in all_series.items():
    plt.plot(xs, ys, label=name)
plt.title("Training & Validation Metrics")
plt.xlabel("Step / Epoch")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



def plot_train_val_loss(event_file, title=None):
    """
    从 single TensorBoard event file 中读取 train/loss 和 val/loss，
    并绘制一张训练/验证 loss 曲线图。
    """
    assert os.path.exists(event_file), f"找不到文件: {event_file}"

    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    # 检查 tag
    scalars = ea.Tags().get("scalars", [])
    assert 'train/loss' in scalars and 'val/loss' in scalars, \
        f"{event_file} 中缺少 train/loss 或 val/loss。可用 tags: {scalars}"

    train_events = ea.Scalars('train/loss')
    val_events = ea.Scalars('val/loss')

    train_steps = [e.step for e in train_events]
    train_vals = [e.value for e in train_events]
    val_steps = [e.step for e in val_events]
    val_vals = [e.value for e in val_events]

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_vals, label='Train Loss')
    plt.plot(val_steps, val_vals, label='Validation Loss')
    plt.title(title or os.path.basename(event_file))
    plt.xlabel('Step/Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_train_val_loss(
    r"D:\TransUnet\trainingrecords_transUnet\log\mydata_transunet_Lovasz\events.out.tfevents.1753868326.DESKTOP-M39MNQU.32204.0",
    title="Trans-unet: Train/Val Loss"
)