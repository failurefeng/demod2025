import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator  # 新增刻度控制库

# 设置全局样式
plt.style.use('seaborn')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("./results/Transformer_puredemod/best/ours/checkpoints/150epoch_training_progress.csv")

# 自动生成Epoch序列
epochs = range(1, len(df)+1)

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
fig.patch.set_facecolor('#F5F5F5')
ax.set_facecolor('white')

# ---- 颜色和样式 ----
train_color = '#cc0000'
valid_color = '#000099'
background_color = '#F5F5F5'
grid_color = '#DDDDDD'

# 绘制曲线（保持原样）
ax.plot(epochs, df["Train_CQ"].values, color=train_color, marker='o', markersize=3, linewidth=1.2, alpha=0.8, label='训练集CQ分数')
ax.plot(epochs, df["Valid_CQ"].values, color=valid_color, marker='o', markersize=3, linewidth=1.2, alpha=0.9, label='验证集CQ分数')

# ---- 纵轴刻度优化 ----
ax.yaxis.set_major_locator(MultipleLocator(1))  # 关键修改：纵轴每1单位一个刻度
ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.9, color=grid_color)

# 保留原有横向网格设置
ax.grid(True, axis='x', linestyle='--', linewidth=0.8, alpha=0.9, color=grid_color)

# ---- 其他设置保持不变 ----
[ax.spines[s].set_visible(False) for s in ['right','top','left','bottom']]
ax.set_title("训练与验证CQ分数变化趋势", fontsize=16, pad=25)
ax.set_xlabel("训练轮次 (Epoch)", fontsize=13, color='#444444')
ax.set_ylabel("CQ分数", fontsize=13, color='#444444')
ax.set_xlim(left=-3, right=len(df)+4)

# 自动计算纵轴范围时确保包含整数刻度
ymin = int(df[["Train_CQ", "Valid_CQ"]].min().min() - 1)
ymax = int(df[["Train_CQ", "Valid_CQ"]].max().max() + 1)
ax.set_ylim(ymin, ymax)

ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#CCCCCC', fontsize=11)

plt.tight_layout()
plt.savefig('training_trend_hd.png', bbox_inches='tight', dpi=600, facecolor=background_color)
plt.show()