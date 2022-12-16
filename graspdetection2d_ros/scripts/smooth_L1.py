#!/usr/bin/env python3
#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

# Print Special math symbols https://wizardforcel.gitbooks.io/matplotlib-user-guide/content/4.6.html
# import matplotlib.font_manager as fm
# # 使用Matplotlib的字体管理器加载中文字体
# my_font=fm.FontProperties(fname="C:\Windows\Fonts\simkai.ttf")


# 支持中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['AR PL UKai CN'] #matplotlib中自带的中文字体
#解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus']=False

x_data = np.arange(-2.0,2.0,0.01)

L1_loss = np.abs(x_data)
L2_loss = np.square(x_data)
smooth_L1_loss = 0.5*np.square(x_data)*(np.abs(x_data)<1) + (np.abs(x_data)-0.5)*(np.abs(x_data)>=1)

x_data_ex = np.arange(-1.0,1.0,0.01)
smooth_L1_loss_ext = (np.abs(x_data_ex)-0.5)*(np.abs(x_data_ex)<=1)




# 指定折线的颜色、线宽和样式
ln2, = plt.plot(x_data, L2_loss, color = 'cyan', linewidth = 2.0, linestyle = '--',label='L2 Loss')
ln1, = plt.plot(x_data, L1_loss, color = 'pink', linewidth = 2.0, linestyle = '-.',label='L1 Loss')
ln3, = plt.plot(x_data, smooth_L1_loss, color = 'green', linewidth = 2.0, linestyle = '-',label='Huber Loss')
ln4, = plt.plot(x_data_ex, smooth_L1_loss_ext, color = 'green', linewidth = 1.0, linestyle = ':')

plt.xlim([-2,2])
plt.ylabel('loss value')
plt.xlabel('$|G_{i}-\^G_{i}|$')  # +'$\^G_i_k$'
plt.grid(linestyle='-.')

plt.scatter(1,0.5,s=80,color='k')
plt.scatter(-1,0.5,s=80,color='k')
ax = plt.gca()
ax.set_aspect(1)

# 调用legend函数设置图例
# plt.legend(loc='lower right', prop=my_font)
plt.legend(loc='upper right')
# 调用show()函数显示图形
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\smoothL1.svg')
plt.show()
