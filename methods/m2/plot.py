import json
import matplotlib.pyplot as plt
import os

# 读取你刚才训练存下来的数据
with open("results/preds_m2/history_seed42.json", "r") as f:
    history = json.load(f)

# 把轮数和 loss 拿出来
epochs = [item["epoch"] for item in history]
losses = [item["loss"] for item in history]

# 画图
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', color='red')
plt.title("M2 1D-CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# 存图
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig_m2_training.png")
print("图画好了！去 figures 文件夹里拿 fig_m2_training.png")