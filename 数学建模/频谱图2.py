import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# —— 中文 & 负号 ——
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
labels = ["正常", "滚动体故障", "内圈故障", "外圈故障"]
def read_single_column_numbers(file_path: Path):
    """从任意分隔/含杂项字符的文本中提取所有数值为一维浮点数组。"""
    if not file_path.exists():
        return None
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
    if not nums:
        return np.array([], dtype=float)
    return np.array(list(map(float, nums)), dtype=float)

# 要绘制的文件
files  = [Path(f"{i}.csv") for i in range(1, 5)]
# labels = [f"{i}.csv" for i in labels]
colors = ["#63b2ee", "#76da91", "#f8cb7f", "#f89588"]  # 易分辨配色
alpha  = 0.8
T = 8.0  # 0~8 s

# 读取全部数据
series = [read_single_column_numbers(p) for p in files]

# —— 统一 y 轴范围（稳健：1%~99% 分位数）——
valid = [s for s in series if isinstance(s, np.ndarray) and s.size > 0]
if valid:
    all_vals = np.concatenate(valid)
    y_low, y_high = np.quantile(all_vals, [0.01, 0.99])  # 稳健范围
    # 如果想用“严格全范围统一”，请改成：
    # y_low, y_high = np.min(all_vals), np.max(all_vals)
    pad = 2.3 * (y_high - y_low + 1e-12)  # 边距
    y_min_glob, y_max_glob = y_low - pad, y_high + pad
else:
    y_min_glob, y_max_glob = None, None  # 兜底

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(11, 6), dpi=120, constrained_layout=True)
axes = axes.ravel()

for ax, y, lab, c in zip(axes, series, labels, colors):
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_title(lab)

    if y is None:
        ax.text(0.5, 0.5, "未找到文件", ha="center", va="center", transform=ax.transAxes, color="gray")
        continue
    if y.size == 0:
        ax.text(0.5, 0.5, "未解析到数值", ha="center", va="center", transform=ax.transAxes, color="gray")
        continue

    # 统一 y 轴
    if y_min_glob is not None:
        ax.set_ylim(y_min_glob, y_max_glob)

    # 时间轴均匀映射到 0~8 s
    t = np.linspace(0, T, len(y), endpoint=True)
    ax.plot(t, y, color=c, linewidth=1.6, alpha=alpha)

# fig.suptitle("四个子图（统一 y 轴）：每图一条频率曲线（0–8 s）", fontsize=13)
plt.show()
