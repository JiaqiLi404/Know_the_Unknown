# @Time : 2024/3/27 20:21
# @Author : Li Jiaqi
# @Description :
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

labels = ['GPT-4', 'LLaMA2', 'Cognition', 'HotpotQA', 'PSQA',"Cognition \n& QA","Cognition \n& PSQA"]
f1s=[81.6,67.2,64.6,66.5,69.6,32.8,85.8]
pos_accs=[79.6,79.3,52.4,77.5,84.8,77.0,79.7]
neg_accs=[83.6,58.3,84.4,58.3,59.0,20.9,93.0]

x = np.arange(len(labels))  # the label locations
width = 0.18  # the width of the bars

fig, ax = plt.subplots(figsize=(6, 4))
rects1 = ax.bar(x - 1.3* width, f1s, width, label='F1 Score',color='salmon')
rects2 = ax.bar(x , pos_accs, width, label='Accuracy of Positive Samples',color='deepskyblue')
rects3 = ax.bar(x + 1.3* width , neg_accs, width, label='Accuracy of Negative Samples',color='lightslategray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('score')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.ylim(50,100)
plt.show()
