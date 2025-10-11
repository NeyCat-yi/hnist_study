import matplotlib.pyplot as plt

labels = ['IR', 'OR', 'N', 'B']
sizes = [40, 77, 4, 40]
explode = (0, 0.1, 0, 0)

fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#f8cb7f', '#f89588', '#7cd6cf', '#9192ab'],
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},  # 白色分割线
    pctdistance=0.7,    # 百分比文本距圆心位置
    labeldistance=1.05  # 标签文本距圆心位置
)

# —— 设置“标签文字”（IR/OR/B/N）的字体和颜色 ——
for txt in texts:
    txt.set_fontsize(12)
    txt.set_color('#333333')
    # 如果需要中文字体，可改为 Microsoft YaHei
    # txt.set_fontfamily('Microsoft YaHei')

# —— 设置“百分比文字”（autopct）的字体和颜色 ——
for at in autotexts:
    at.set_fontsize(8)
    at.set_color("#FFFFFF")
    at.set_fontweight('bold')
    at.set_fontfamily('Microsoft YaHei')

# —— 图例 ——（使用 wedges 作为句柄；显示“名称（数量）”）
legend_labels = [f"{lab} ({val})" for lab, val in zip(labels, sizes)]
leg = ax.legend(
    wedges, legend_labels,
    # title="Legend",
    loc="center left",           # 图例位置
    bbox_to_anchor=(1.02, 0.5),  # 挪到图外右侧，避免遮挡
    frameon=False,               # 无边框更简洁
    handlelength=1.2,
    handletextpad=0.6
)
# 若想两列排布：把 ncol=2 加上
# leg = ax.legend(wedges, legend_labels, ncol=2, ...)

ax.axis('equal')  # 保持正圆
plt.tight_layout()
plt.show()
