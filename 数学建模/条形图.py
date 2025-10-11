import matplotlib.pyplot as plt


# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 8))

names = ['P62801', 'P11648', 'P11668', 'P40106', 'P66666', 'P23815', 'P03461', 'P11148', 
         'P01929', 'P57158', 'P14588', 'P13145', 'P84958', 'P73482', 'P70730', 'P06349'] # top16
counts = [2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 1]
# for i in range(10):
#     counts.append(1) 
# bar_labels = ['red']
# bar_colors = ['#63b2ee ']

ax.bar(names, counts, color='#7cd6cf', width=0.5)
ax.tick_params(axis='x', labelsize=14)  # 减小x轴标签字体，避免重叠
plt.xticks(rotation=45, ha='right')    # 旋转标签45度，右对齐，防止重叠
ax.set_ylabel('未通过次数', fontsize=16)
# ax.set_title('Fruit supply by kind and color')
# ax.legend(title='Fruit color')

plt.show()