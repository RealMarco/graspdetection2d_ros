import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from io import StringIO

plt.figure(1)
#font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
plt.plot([0,44],[0.88,0.88],color='k',linestyle='--',linewidth=0.4)
plt.plot([0,44],[0.95,0.95],color='k',linestyle='--',linewidth=0.4)

net0 = pd.read_csv('run-4_Adam-tag-loss_IOU.csv', usecols=['Step', 'Value'])
plt.plot(net0.Step, net0.Value, lw=1.5, label='Adam(lr=0.001)', color='pink',linestyle='--')

net1 = pd.read_csv('run-4_Adam_SGD-tag-loss_IOU.csv', usecols=['Step', 'Value'])
plt.plot(net1.Step, net1.Value, lw=1.5, label='Adam(lr=0.001)+SGD(lr=0.0001)', color='red')



title1 = '4th Test Accuravy - Adam & Adam+SGD - OW'
plt.title(title1)
plt.legend(loc='lower right')
plt.xlim((0,44))
plt.xlabel('Number of Epochs') #fontproperties=font
plt.ylabel('Test Accuracy')
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\%s.svg' % (title1))
plt.show()

#-----------------
plt.figure(2)

plt.plot([0,44],[0.075,0.075],color='k',linestyle='--',linewidth=0.4)

net2 = pd.read_csv('run-4_Adam-tag-loss_train_loss.csv', usecols=['Step', 'Value'])
plt.plot(net2.Step, net2.Value, lw=1.5, label='Adam(lr=0.001)', color='cyan',linestyle='--')

net3 = pd.read_csv('run-4_Adam_SGD-tag-loss_train_loss.csv', usecols=['Step', 'Value'])
plt.plot(net3.Step, net3.Value, lw=1.5, label='Adam(lr=0.001)+SGD(lr=0.0001)', color='blue')


title2 = '4th Training Loss - Adam & Adam+SGD - OW'
plt.title(title2)
plt.legend(loc='upper right')
plt.xlim((0,44))
plt.xlabel('Number of Epochs') #fontproperties=font
plt.ylabel('Training Loss')
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\%s.svg' % (title2))
plt.show()
