import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from io import StringIO

plt.figure(1)
#font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
plt.plot([0,29],[0.90,0.90],color='k',linestyle='--',linewidth=0.4)
plt.plot([0,29],[0.80,0.80],color='k',linestyle='--',linewidth=0.4)

net0 = pd.read_csv(r'F:\OD and GD Based on DL\Results Samples\run-2_without_augment-tag-loss_IOU.csv', usecols=['Step', 'Value'])
plt.plot(net0.Step, net0.Value, lw=1.5, label='Without Augmentation', color='pink',linestyle='--')

net1 = pd.read_csv(r'F:\OD and GD Based on DL\Results Samples\run-2_with_augment-tag-loss_IOU.csv', usecols=['Step', 'Value'])
plt.plot(net1.Step, net1.Value, lw=1.5, label='With Augmentation', color='red')


title1= '2nd Test Accuracy - Augmentation - IW'
plt.title(title1)
plt.legend(loc='lower right')
plt.xlim((0,30))
plt.xlabel('Number of Epochs') #fontproperties=font
plt.ylabel('Test Accuracy')
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\%s.svg' % (title1))
plt.show()

#-----------

plt.figure(2)

net2 = pd.read_csv(r"F:\OD and GD Based on DL\Results Samples\run-2_without_augment-tag-loss_train_loss.csv", usecols=['Step', 'Value'])
plt.plot(net2.Step, net2.Value, lw=1.5, label='Without Augmentation', color='cyan',linestyle='--')

net3 = pd.read_csv(r"F:\OD and GD Based on DL\Results Samples\run-2_with_augment-tag-loss_train_loss.csv", usecols=['Step', 'Value'])
plt.plot(net3.Step, net3.Value, lw=1.5, label='With Augmentation', color='blue')


title1= '2nd Training Loss - Augmentation - IW'
plt.title(title1)
plt.legend(loc='upper right')
plt.xlim((0,30))
plt.xlabel('Number of Epochs') #fontproperties=font
plt.ylabel('Training Loss')
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\%s.svg' % (title1))
plt.show()
