from turtle import color
import matplotlib.pyplot as plt 
import os
import pandas as pd
from pandas.core.algorithms import mode
import json
import seaborn as sns
import numpy as np


# result_df_AMM = pd.read_csv('AMM_07_03_22_latest.csv')
result_df_MAM = pd.read_csv('MAM_07_03_22_latest.csv')
result_df_AMM = result_df_MAM

fig, ax1 = plt.subplots()
sns.set_palette("pastel")
ax1 = sns.lineplot(data=result_df_AMM, x = 'N' , y ='Optical Recieved Power(dBm)', hue= 'SOMA (dBm)' , linewidth=3)
ax1.set_ylabel('Optical Received Power (dBm)')
ax1.yaxis.label.set_fontsize(18)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_fontweight('bold')
ax1.yaxis.label.set_fontweight('bold')
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_xlim(0,43)
ax1.set_ylim(-24,-4)
ax1.get_legend().remove()


ax2 = ax1.twinx()
ax2 = sns.lineplot(data=result_df_AMM,x='N', y = 'Products Supported Precision', hue='SOMA (dBm)', palette=sns.color_palette('bright', n_colors=3), legend= False)
ax2.set_ylabel('Supported Precision (Bits)')
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_fontweight('bold')
ax2.set_ylim(0,10)

ax2.tick_params(axis='y', labelsize=14)

ax2.yaxis.set_ticks(np.arange(0, 11, 1))


ax3 = ax1.twinx()
ax3.spines.right.set_position(("axes", 1))
ax3 = sns.lineplot(data=result_df_AMM,x='N', y = 'Sum of Products Supported Precision', hue='SOMA (dBm)', linestyle='--', palette=sns.color_palette('bright', n_colors=3), legend= False)
ax3.axes.get_yaxis().set_visible(False)
ax3.yaxis.set_ticks(np.arange(0, 11, 1))
# plt.legend(title='SOMA (dBm)', fontsize='10', title_fontsize='14')
plt.rcParams.update({'font.family':'Times New Roman'})
plt.savefig('Plots/Sample.png')
plt.show()
# print(result_df)