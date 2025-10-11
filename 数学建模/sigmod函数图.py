import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 定义Sigmoid函数
def sigmoid(x):
    """Sigmoid函数表达式：f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

# 生成x数据范围
x = np.linspace(-10, 10, 1000)  # 从-10到10生成1000个点
y = sigmoid(x)  # 计算对应的y值

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制Sigmoid曲线
plt.plot(x, y, color='#3498db', linewidth=2)

# 绘制水平渐近线
# plt.axhline(y=1, color='#95a5a6', linestyle='--', alpha=0.7)
# plt.axhline(y=0.5, color='#95a5a6', linestyle='--', alpha=0.7)
# plt.axhline(y=0, color='#95a5a6', linestyle='--', alpha=0.7)

# 绘制原点参考线
plt.axvline(x=0, color='#95a5a6', alpha=0.7)

# 添加标题和标签
plt.title('Sigmoid函数图像', fontsize=15)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('σ(x) = 1 / (1 + e^(-x))', fontsize=12)

# 设置坐标轴范围
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)

# 添加网格
plt.grid(alpha=0.3)

# 显示图像
plt.show()
    