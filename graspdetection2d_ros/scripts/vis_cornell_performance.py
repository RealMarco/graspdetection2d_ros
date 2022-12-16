import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from io import StringIO


plt.figure(1)
plt.xscale('log')

net0 = pd.read_csv("F:\OD and GD Based on DL\Results Samples\OW_performance_on_cornell_dataset.csv", usecols=['Algorithms','Accuracy','Speed'])
for i in range(len(net0.Accuracy)):
    if net0.Algorithms[i] =='GR-ConvNet':
        plt.text(net0.Speed[i],net0.Accuracy[i]+2, s='(%.1f,%.1f)'%(net0.Speed[i],net0.Accuracy[i]),ha='center',va='top',fontdict=dict(color='k'))
        plt.scatter(net0.Speed[i],net0.Accuracy[i], label = net0.Algorithms[i], lw=5, marker='^',color='k')
    else:
        plt.scatter(net0.Speed[i],net0.Accuracy[i], label = net0.Algorithms[i], lw=3, marker='o')

title1= 'Performance on Cornell Dataset - OW'
plt.title(title1)
plt.ylabel('Test Accuracy for Object-wise (%)')
plt.xlabel('Prediction Speed(ms/image)')
plt.legend(loc=0)
ax=plt.gca()
ax.invert_xaxis()
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\%s.svg' % (title1))
plt.show()

#------------
plt.figure(2)
plt.xscale('log')

net1 = pd.read_csv("F:\OD and GD Based on DL\Results Samples\IW_performance_on_cornell_dataset.csv", usecols=['Algorithms','Accuracy','Speed'])
for i in range(len(net1.Accuracy)):
    if net1.Algorithms[i] =='GR-ConvNet':
        plt.text(net1.Speed[i],net1.Accuracy[i]+2, s='(%.1f,%.1f)'%(net1.Speed[i],net1.Accuracy[i]),ha='center',va='top',fontdict=dict(color='k'))
        plt.scatter(net1.Speed[i],net1.Accuracy[i], label = net1.Algorithms[i], lw=5, marker='^',color='k')
    else:
        plt.scatter(net1.Speed[i],net1.Accuracy[i], label = net1.Algorithms[i], lw=3, marker='o')

title1= 'Performance on Cornell Dataset - IW'
plt.title(title1)
plt.ylabel('Test Accuracy for Image-wise (%)')
plt.xlabel('Prediction Speed(ms/image)')
plt.legend(loc=0)
ax=plt.gca()
ax.invert_xaxis()
plt.savefig(r'F:\OD and GD Based on DL\Results Samples\%s.svg' % (title1))
plt.show()
