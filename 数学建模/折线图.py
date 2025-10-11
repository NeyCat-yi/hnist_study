import matplotlib.pyplot as plt
import numpy as np


# 设置中文字体为微软雅黑，确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 解决负号显示问题（避免负号变成方块）
plt.rcParams["axes.unicode_minus"] = False

# Data for plotting
name = ['P70730', 'P84958', 'P62801', 'P66666', 'P73482', 'P57158', 'P01929', 'P14588', 'P23815', 'P03461', 'P06349', 'P11148', 'P11648', 'P11668', 'P13145', 'P40106', 'P36274', 'P35453', 'P33579', 'P31942', 'P25822', 'P46254', 'P42983', 'P62063', 'P59290', 'P55755', 'P49524', 'P64683', 'P74368', 'P74368', 'P43939', 'P75094', 'P76668', 'P81023', 'P78709', 'P83179', 'P84041', 'P86371', 'P86530', 'P87006', 'P87327', 'P87874', 'P88206', 'P89885', 'P91385', 'P93663', 'P97778']
#count = [3, 2, 2, 2, 2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
count = [3,2,2,2,2,2]
for i in range(41):
    count.append(1) 
fig, ax = plt.subplots()
ax.plot(name, count)

ax.set(ylabel='未通过次数')
# ax.set(xlabel='姓名', ylabel='未通过次数',
#        title='About as simple as it gets, folks')


# ax.grid()

# fig.savefig("test.png")
plt.show()