import matplotlib.pyplot as plt

# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots()

fruits = ['IR', 'OR', 'B', 'N']
counts = [5, 4, 5, 2]
bar_colors = ['#B5C6E7', '#DFEED5', '#FEF0C7', "#91d8d2"]
bar_labels = ['内圈故障', '外圈故障', '滚动体故障', '正常']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('预测数量', fontsize=16)
ax.set_xlabel('类别', fontsize=16)
ax.legend()

plt.show()