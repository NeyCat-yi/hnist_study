# 导入matplotlib绘图库和numpy数值计算库
import matplotlib.pyplot as plt
import numpy as np

# 从matplotlib.patches导入Patch类，用于创建图例元素
from matplotlib.patches import Patch
# 从matplotlib.transforms导入Bbox类（当前代码中注释部分使用，用于图像裁剪）
from matplotlib.transforms import Bbox

# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False

# 数据准备：定义年份和对应数值
years = ['2024', '2023', '2022', '2021', '2020', '2019']  # 年份标签
values = [40.85, 36.85, 16.10, 25.33, 21.67, 35.79]  # 对应数值（单位：亿人次）
colors = ['#1f77b4', '#00ced1', '#ffd700', '#ff7f0e', '#e31a1c', '#d8bfff']  # 每个扇形的颜色

# 计算绘图所需参数
total_val = sum(values)  # 计算数值总和（用于分配扇形角度）
max_val = max(values)    # 找到最大值（用于按比例计算半径）
min_radius = 0.4         # 最小半径（基础半径值）
# 计算每个扇形的半径：与数值成正比，范围在0.4~1.0之间
radii = [min_radius + (v / max_val) * 0.6 for v in values]
# 计算每个扇形的角度（度）：按数值占比分配180度（上半圆）
angles_deg = [v / total_val * 180 for v in values]

# 将角度转换为弧度（matplotlib极坐标使用弧度）
angles_rad = np.radians(angles_deg)
current_angle = np.radians(0)  # 当前角度初始值（从0弧度开始绘制，对应右侧水平方向）

# 创建画布和极坐标子图：figsize设置画布大小为10x8英寸
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
ax.set_facecolor('white')  # 设置子图背景色为白色

# 循环绘制每个扇形
for i in range(len(values)):
    angle = angles_rad[i]  # 当前扇形的角度（弧度）
    radius = radii[i]      # 当前扇形的半径
    
    # 绘制扇形（极坐标下的柱状图）
    ax.bar(
        current_angle + angle/2,  # 扇形中心角度位置
        radius,                   # 扇形半径（高度）
        width=angle,              # 扇形角度宽度
        bottom=0,                 # 从原点（0）开始绘制
        color=colors[i],          # 扇形填充色
        edgecolor='white',        # 扇形边缘色（白色，区分相邻扇形）
        alpha=0.9                 # 透明度（0.9表示轻微透明）
    )
    
    # 在扇形内部添加数值标签
    ax.text(
        current_angle + angle/2,  # 文本x坐标（扇形中心角度）
        radius * 0.8,              # 文本y坐标（半径的80%位置，扇形中间）
        f'{values[i]:.1f}',       # 显示的文本内容（保留1位小数）
        ha='center', va='center', # 水平和垂直都居中对齐
        fontsize=18,              # 字体大小12
        weight=1000,               # 字体粗细（900对应极粗）
        color='white'             # 文本颜色白色（与深色扇形对比）
    )
    
    # 在扇形外侧添加年份标签
    ax.text(
        current_angle + angle/2,  # 文本x坐标（扇形中心角度）
        radius + 0.1,              # 文本y坐标（扇形外侧，半径+0.1）
        years[i],                  # 显示的年份文本
        ha='center', va='center', # 水平和垂直都居中对齐
        fontsize=18,              # 字体大小12
        color='#333333'           # 文本颜色深灰色
    ) 
    
    current_angle += angle  # 更新当前角度，准备绘制下一个扇形

# 极坐标显示范围设置
ax.set_thetalim(np.radians(0), np.radians(180))  # 仅显示0~180度（上半圆）
ax.set_ylim(0, 1.2)  # 设置半径范围0~1.2（留出标签空间）
ax.axis('off')  # 隐藏极坐标的坐标轴和刻度

# 添加颜色注解（图例）
# 创建图例元素列表：每个元素对应一个年份的颜色块
legend_elements = [Patch(facecolor=color, edgecolor='white', label=year) 
                   for color, year in zip(colors, years)]
# 绘制图例
ax.legend(handles=legend_elements, 
          title="年份",               # 图例标题
          loc='upper left',           # 图例位置（左上角）
          bbox_to_anchor=(0.05, 0.95),# 图例偏移位置（左5%，上95%）
          ncol=3,                     # 图例分3列显示
          title_fontsize=19,          # 标题字体大小15
          fontsize=16)                # 图例文字大小12

# 在右上角添加单位说明文本
plt.text(
    0.98, 0.90,  # 文本位置（相对坐标：x=0.98右侧，y=0.90偏上）
    '单位：亿人次',  # 显示的文本内容
    ha='right', va='top',  # 右对齐、顶部对齐
    fontsize=25,          # 字体大小18
    color="#161616",      # 文本颜色深黑色
    transform=ax.transAxes # 使用相对坐标系（0~1范围）
)

# 以下为注释掉的图像裁剪保存代码
# 核心修复：用Bbox类创建正确的裁剪范围（保留上半部分）
# 1. 获取画布原始边界（单位：英寸）
# original_bbox = fig.bbox_inches
# 2. 计算上半部分的边界：左、下（从画布中间开始）、右、上
# 格式：Bbox.from_bounds(左边界, 下边界, 宽度, 高度)
# cropped_bbox = Bbox.from_bounds(
#     original_bbox.x0,  # 左边界不变（与原画布一致）
#     original_bbox.y0 + original_bbox.height / 1.2,  # 下边界上移到画布中间（裁剪下半）
#     original_bbox.width,  # 宽度不变（与原画布一致）
#     original_bbox.height / 2  # 高度为原画布的1/2（仅保留上半）
# )

# 保存图像（传入正确的Bbox对象，避免报错）
# plt.savefig(
#     '图1_裁剪_方法三（修复版）.png',
#     dpi=300,
#     bbox_inches=cropped_bbox,  # 现在是Bbox对象，可正常识别
#     facecolor='white'
# )

# 自动调整布局（避免元素重叠）
plt.tight_layout()
# 显示图像
plt.show()