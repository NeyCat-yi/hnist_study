import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------- 数据 ----------------
years = [2020, 2021, 2022, 2023, 2024]

# 第一条：
data1 = [6188.42, 6477.10, 6789.82, 7499.07, 7918.80] # 十种有色金属产量

# 第二条：
data2 = [1002.51, 1045.67, 1111.53, 1325.57, 1364.40] # 精炼铜产量

data3 = [3708.04, 3850.32, 4014.43, 4197.97, 4400.50] # 原铝产量

data4 = [7313.19, 7747.54, 8186.18, 8251.17, 8342.16] # 氧化铝产量

# ---------------- 字体与负号设置（自动适配常见中文字体） ----------------
plt.rcParams["font.family"] = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 画图 ----------------
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)

# 两条折线（不手动指定颜色，使用 matplotlib 默认调色）
ax.plot(years, data1, linewidth=2.2, marker='o', markersize=5, label="十种有色金属产量(万吨)")
ax.plot(years, data2, linewidth=2.0, marker='s', markersize=5, label="精炼铜产量(万吨)")
ax.plot(years, data3, linewidth=2.0, marker='s', markersize=5, label="原铝产量(万吨)")
# ax.plot(years, data4, linewidth=2.0, marker='s', markersize=5, label="氧化铝产量(万吨)")

# 坐标轴标题与主标题
ax.set_xlabel("年份")
ax.set_ylabel("产量（万吨）")
#ax.set_title("test")

# x 轴显示全部年份
ax.set_xticks(years)

# 网格（虚线，半透明）
ax.grid(True, which="both", linestyle='--', linewidth=0.8, alpha=0.6)

# y 轴千分位与两位小数格式
def yfmt(x, pos):
    return f"{x:,.2f}"
ax.yaxis.set_major_formatter(FuncFormatter(yfmt))

# 根据两条曲线的整体范围自动留白
y_all = data1 + data2
y_min, y_max = min(y_all), max(y_all)
y_pad = (y_max - y_min) * 0.08 if (y_max - y_min) > 0 else 100
ax.set_ylim(y_min - y_pad, y_max + y_pad)

# 图例
ax.legend(loc="best", frameon=False)

# 末端数据标注（仅标注最后一个点）
def annotate_last(x_list, y_list, label):
    x, y = x_list[-1], y_list[-1]
    ax.annotate(f"{label}: {y:,.2f}",
                xy=(x, y),
                xytext=(8, 8),
                textcoords="offset points")

# annotate_last(years, data1, "")
# annotate_last(years, data2, "")

# 紧凑布局与保存
fig.tight_layout()
fig.savefig("折线图.png")
plt.show()
