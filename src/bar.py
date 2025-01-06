import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() 
# 数据
x = ['Color_hard', 'Video_easy', 'Video_hard']
SAC = [1.9522206, 3.2942472, 5.3169312]
SAC_C= [1.0973594, 0.25811297, 0.7150348]
SVEA = [1.16177, 0.34435263, 1.2995937]
SGQN = [1.6687645, 0.3996746, 1.2787015]
SCPL = [0.44652748, 0.028641868, 0.18164024]

# plt.rc('font',family='Times New Roman') 
# 设置条形的宽度
bar_width = 0.15
linewidth = 1
hline_width = 0.05


rSAC = [411, 300, 145]
rSAC_C= [808, 874, 733]
rSVEA = [895, 848, 467]
rSGQN = [870, 906, 747]
rSCPL = [930, 930, 853]

# 创建位置信息
bar_positions = np.arange(len(x))

# 设置颜色
colors = ['#0071bc', '#d85218', '#70ad47', '#954f72', '#ffd700']

fig, ax1 = plt.subplots()
ax1.grid(True)
# 绘制条形图
ax1.bar(bar_positions, SAC, width=bar_width, label='SAC', color=colors[0], edgecolor='none')
ax1.bar(bar_positions + bar_width, SVEA, width=bar_width, label='SVEA', color=colors[1], edgecolor='none')
ax1.bar(bar_positions + 2 * bar_width, SGQN, width=bar_width, label='SGQN', color=colors[2], edgecolor='none')
ax1.bar(bar_positions + 3 * bar_width, SAC_C, width=bar_width, label='SCPL w/o policy consisenty', color=colors[3], edgecolor='none')
ax1.bar(bar_positions + 4 * bar_width, SCPL, width=bar_width, label='SCPL (ours)', color=colors[4], edgecolor='none')
ax1.set_ylabel('KL Divergence', fontsize=14)
ax1.set_ylim(0, 6)

ax2 = ax1.twinx()
ax2.grid(False)
ax2.vlines(bar_positions, ymin=0, ymax=rSAC, color="black", linewidth=linewidth)
ax2.vlines(bar_positions + 1 * bar_width, ymin=0, ymax=rSVEA, color="black", linewidth=linewidth)
ax2.vlines(bar_positions + 2 * bar_width, ymin=0, ymax=rSGQN, color="black", linewidth=linewidth)
ax2.vlines(bar_positions + 3 * bar_width, ymin=0, ymax=rSAC_C, color="black", linewidth=linewidth)
ax2.vlines(bar_positions + 4 * bar_width, ymin=0, ymax=rSCPL, color="black", linewidth=linewidth)
ax2.set_ylabel('Return', fontsize=14)
ax2.set_ylim(0, 1700)

plt.hlines(rSAC, xmin= bar_positions  - hline_width, xmax= bar_positions + hline_width, color="black", linewidth=linewidth)
plt.hlines(rSVEA, xmin= bar_positions + bar_width - hline_width, xmax= bar_positions + bar_width + hline_width, color="black", linewidth=linewidth)
plt.hlines(rSGQN, xmin= bar_positions + 2*bar_width - hline_width, xmax= bar_positions + 2*bar_width + hline_width, color="black", linewidth=linewidth)
plt.hlines(rSAC_C, xmin= bar_positions + 3*bar_width - hline_width, xmax= bar_positions + 3*bar_width + hline_width, color="black", linewidth=linewidth)
plt.hlines(rSCPL, xmin= bar_positions + 4*bar_width - hline_width, xmax= bar_positions + 4*bar_width + hline_width, color="black", linewidth=linewidth)

# 添加标题和标签，并增大字体大小
# plt.title('')
# plt.xlabel('Metrics', fontsize=12)
# plt.ylabel()
# 添加标题和标签，并增大字体大小
plt.xlabel('Metrics', fontsize=14)
ax1.set_ylabel('KL Divergence', fontsize=14)
ax2.set_ylabel('Return', fontsize=14)
plt.suptitle('Comparison', fontsize=14)

# 设置刻度标签，并增大字体大小
plt.xticks(bar_positions + 1.5 * bar_width, x, fontsize=18)
plt.yticks(fontsize=14)

# 添加图例，并增大字体大小
ax1.legend(fontsize=14)

# # 去除上边界、右边界和下边界的线
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# # 调整刻度标签的字体大小
# plt.xticks(bar_positions + 1.5 * bar_width, x, fontsize=18)
# plt.yticks(fontsize=18)

# 去除上边界、右边界和下边界的线
for ax in (ax1, ax2):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

# 调整刻度标签的字体大小
for ax in (ax1, ax2):
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

# 设置图例放在左上角，去掉图例边框，图例框内透明
legend = ax1.legend(loc='upper left', prop={'size': 14})
legend.get_frame().set_linewidth(0)
legend.get_frame().set_alpha(0)

plt.tight_layout()
# 调整图形比例为1:1
fig = plt.gcf()
fig.set_size_inches(7, 5)

# 保存图像
plt.savefig('/root/deeplearningnew/sun/rl_generalization/dmcontrol-generalization-benchmark-main/draw_new/bar_chart.png')
plt.savefig('/root/deeplearningnew/sun/rl_generalization/dmcontrol-generalization-benchmark-main/draw_new/bar_chart.pdf', dpi=600)

