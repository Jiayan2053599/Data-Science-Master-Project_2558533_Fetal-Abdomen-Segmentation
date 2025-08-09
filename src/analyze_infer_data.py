from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 替换为你的文件路径
event_file = r"D:\nnUNet\results\Dataset300_ACOptimalSuboptimal2D(data_split_correct)\MyTrainer__nnUNetPlans__2d\fold_0\tb_logs\events.out.tfevents.1754321972.ip-172-31-30-239.eu-west-2.compute.internal.257007.0"
# 加载日志
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 所有可用 scalar
print("Scalars:", ea.Tags()["scalars"])

# 读取指定 scalar 并绘图
def plot_scalar(tag, label):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=label)

# 画图
plt.figure(figsize=(10, 6))
plot_scalar("train/loss", "Train Loss")
plot_scalar("val/loss", "Validation Loss")
plot_scalar("val/mean_dice", "Mean Dice")
plot_scalar("val/dice_class_0", "Dice Class 0")
plot_scalar("val/dice_class_1", "Dice Class 1")
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Training and Validation Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

