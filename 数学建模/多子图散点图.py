import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
def plot_correlation(ax, correlation_type=0):
    """
    生成不同相关性的散点图
    
    参数:
        ax: 子图对象
        correlation_type: 相关性类型
            'positive' - 正相关性
            'negative' - 负相关性
            'nonlinear' - 非线性关系
            'none' - 无相关性
    """
    n = 35  # 散点数量
    x_min, x_max = 0, 12  # x 的取值范围
    disturbance = 1.5  # 扰动幅度
    
    # 数据集中比例和起始点
    concentrate_ratio = 0.9
    concentrate_start = 6
    
    # 生成x值（保持原分布特征）
    mask = np.random.choice([True, False], size=n, p=[concentrate_ratio, 1-concentrate_ratio])
    x = np.where(
        mask,
        np.random.uniform(low=concentrate_start, high=x_max, size=n),
        np.random.uniform(low=x_min, high=concentrate_start, size=n)
    )
    
    # 根据相关性类型生成不同的y值
    if correlation_type == 0:
        # 正相关性: y随x增大而增大
        y = x + 1 + (np.random.rand(n) - 0.5) * 2 * disturbance
        title = "正相关性"
        color = '#82B0D2'
        
    elif correlation_type == 1:
        # 负相关性: y随x增大而减小
        y = -x + 13 + (np.random.rand(n) - 0.5) * 2 * disturbance  # 13是为了让数据在合理范围内
        title = "负相关性"
        color = '#FF9999'
        
    elif correlation_type == 2:
        # 非线性关系: 使用二次函数
        y = 0.1 * (x - 6)**2 + 3 + (np.random.rand(n) - 0.5) * 2 * disturbance
        title = "非线性关系"
        color = '#99FF99'
        
    elif correlation_type == 3:
        # 无相关性: y与x无关，随机分布
        y = np.random.uniform(low=1, high=13, size=n)  # 在合理范围内随机分布
        title = "无相关性"
        color = '#FFCC99'
    
    # 生成点的大小
    scale = 100 * np.random.rand(n) + 20
    
    # 绘制散点图
    ax.scatter(
        x, y,
        c=color,
        s=scale,
        alpha=0.6,
        edgecolors='none'
    )
    

# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig, axs = plt.subplots(2, 2)

axs[0, 0].set_title('(1)正线性相关')

axs[0, 1].set_title('(2)负线性相关')

axs[1, 0].set_title('(3)非线性相关')

axs[1, 1].set_title('(4)无相关')

count = 0
for ax in axs.flat:
    ax.spines['top'].set_visible(False)    # 隐藏顶部边框
    ax.spines['right'].set_visible(False)  # 隐藏右边边框
    # 关闭x轴主刻度
    ax.set_xticks([])
    # 关闭y轴主刻度
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    # ========= 添加坐标轴箭头 =========
    # 在 x 轴和 y 轴的末端添加箭头
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)  # X 轴箭头
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)  # Y 轴箭头
    plot_correlation(ax, count)
    count += 1
plt.show()