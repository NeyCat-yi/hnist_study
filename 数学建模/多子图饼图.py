import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False
# 定义颜色：通过（浅蓝绿）、未通过（浅红）
colors = ['#8ECFC9', '#FA7F6F']

# 创建 1 行 3 列的子图布局，调整整体大小
fig, axes = plt.subplots(1, 3, figsize=(15, 6), subplot_kw=dict(aspect="equal"))

# 三个饼图的数据（可以根据实际需求修改）
recipe1 = ["31 优秀", "117 不合格"]
recipe2 = ["51 优秀", "135 不合格"]
recipe3 = ["52 优秀", "257 不合格"]
all_recipes = [recipe1, recipe2, recipe3]

# 每个子图的标题
titles = ["4月", "5月", "6月"]

def func(pct, allvals):
    """计算百分比和绝对值的标签格式"""
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d})"

# 遍历每个子图并绘制饼图
for i, (ax, recipe, title) in enumerate(zip(axes, all_recipes, titles)):
    # 提取数据和标签
    data = [float(x.split()[0]) for x in recipe]
    ingredients = [x.split()[-1] for x in recipe]
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        data,
        autopct=lambda pct: func(pct, data),
        textprops=dict(color="w"),
        colors=colors,
    )
    # 单独设置扇形边缘（兼容旧版本的关键）
    for wedge in wedges:
        wedge.set_edgecolor('white')  # 边缘颜色
        wedge.set_linewidth(2)        # 边缘宽度

    # 设置子图标题
    ax.set_title(title, fontsize=16)
    
    # 只在第一个子图显示图例（避免重复）
    if i == 2:
        ax.legend(
            wedges, 
            ingredients,
            title="结果",
            loc='upper right',
            fontsize=14, 
            title_fontsize=16,
            frameon=True,              # 显示图例边框（可选）
            bbox_to_anchor=(1.1, 1)  # x=1.1 表示向右移动10%子图宽度，y=1保持原垂直位置
        )
    
    # 设置数值标签样式
    plt.setp(autotexts, size=14, weight="bold")

# 调整子图间距，避免重叠
plt.tight_layout()

plt.show()
    