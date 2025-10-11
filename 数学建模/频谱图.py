import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# —— 显示中文 & 负号 ——
plt.rcParams["font.family"] = ["Microsoft YaHei"]  # 若本机无该字体会回退
plt.rcParams["axes.unicode_minus"] = False

def read_single_column_numbers(file_path: str):
    """从任意分隔/含杂项字符的文本中，提取所有数值为一维浮点数组。"""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
    arr = np.array(list(map(float, nums)), dtype=float)
    return arr

# 读取两份频率数据（单位按你的数据为准，常见 Hz）
freq1 = read_single_column_numbers("3.csv")
freq2 = read_single_column_numbers("2.csv")

# 生成各自的时间轴：把各自长度均匀映射到 0~8 秒
T = 8.0
t1 = np.linspace(0, T, len(freq1), endpoint=True)
t2 = np.linspace(0, T, len(freq2), endpoint=True)

# 绘图
# plt.figure(figsize=(10, 4), dpi=120)
# plt.plot(t1, freq1, linewidth=1.4, label="1.csv")
# plt.plot(t2, freq2, linewidth=1.4, label="2.csv")
# plt.xlim(0, T)
# plt.xlabel("时间 (s)")
# plt.ylabel("频率 (Hz)")
# plt.title("两份数据的频率-时间对比")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()
plt.figure(figsize=(10, 4), dpi=120)

plt.plot(t2, freq2, linewidth=1.6, alpha=0.9, color="#999FFF", label="原数据")  # 朱橙
plt.plot(t1, freq1, linewidth=1.6, alpha=0.6, color="#FA7F6F", label="去噪后数据")  # 蓝


plt.xlim(0, T)

# 关键：放大 y 轴范围
all_vals = np.concatenate([freq1, freq2])
m = np.median(all_vals)                 # 中心
half = 0.5 * (np.max(all_vals) - np.min(all_vals))
expand = 1.5                           # ← 调大/调小这个数
plt.ylim(m - expand*half, m + expand*half)

plt.xlabel("时间 (s)")
plt.ylabel("频率 (Hz)")
# plt.title("两份数据的频率-时间对比")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()