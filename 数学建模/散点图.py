import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)


fig, ax = plt.subplots()

# 绘制y=x直线
ax.plot([0, 2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11, 13], color='#999999', linewidth=1)

for color in ['tab:"blue"']:  # 颜色列表，这里只包含一种蓝色 而且没用上，可以用多种绘制很多颜色的散点图
    # 生成数据
    n = 60  # 散点数量

    x_min, x_max = 0, 12 # x 的取值范围
    disturbance = 1.5 # 散点偏离 y=x 的扰动幅度（值越大越分散，1.5 适合 0~12 范围）

    concentrate_ratio = 0.9  # 80%的点集中在6~12
    concentrate_start = 6    # 集中区间起始值

    # 核心：按比例生成不同区间的x值
    # 1. 先生成判断数组：80%为True（集中区间），20%为False（低区间）
    mask = np.random.choice([True, False], size=n, p=[concentrate_ratio, 1-concentrate_ratio])

    x = np.where(
        mask,
        np.random.uniform(low=concentrate_start, high=x_max, size=n),  # 6~12
        np.random.uniform(low=x_min, high=concentrate_start, size=n)   # 0~6
    )

    # 在y=x基础上添加微小扰动，使点分布在直线附近
    # 扰动范围控制在±0.1，可根据需要调整（值越小越靠近直线）
    y = x + 1 + (np.random.rand (n) - 0.5) * 2 * disturbance # (rand-0.5)*2 转为 ±1 范围，再乘扰动幅度

    # 生成点的大小（随机值）
    scale = 100 * np.random.rand(n) + 50  # 大小在50-150之间
    
    # 绘制散点图
    ax.scatter(
        x, y,  # 点的横纵坐标
        c='#82B0D2',  # 点的颜色（这里固定为'tab:blue'，一种蓝色）
        s=scale,  # 点的大小（由scale数组决定，每个点大小不同）
        label='#82B0D2',  # 图例标签（显示为颜色名称）
        alpha=0.6,  # 透明度（0.3表示半透明）
        edgecolors='none'  # 点的边缘颜色（'none'表示无边缘）
    )

# ax.legend() # 图例
# ax.grid(True) # 网格
# 核心：隐藏顶部和右边的边框（坐标轴脊线）
ax.spines['top'].set_visible(False)    # 隐藏顶部边框
ax.spines['right'].set_visible(False)  # 隐藏右边边框
ax.spines[["left", "bottom"]].set_position(("data", 0))
# ========= 添加坐标轴箭头 =========
# 在 x 轴和 y 轴的末端添加箭头
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  # X 轴箭头
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)  # Y 轴箭头


ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()