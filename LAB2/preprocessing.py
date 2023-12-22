import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

df1 = pd.read_csv(r'/Users/liusendong/Desktop/original/cpc.csv', encoding="utf_8_sig")

#读取第一个文件

df2 = pd.read_csv(r'/Users/liusendong/Desktop/original/cpm.csv', encoding="utf_8_sig")

#读取第二个文件

outfile = pd.merge(df1, df2)

#文件合并 left_on左侧DataFrame中的列或索引级别用作键。right_on 右侧

outfile.to_csv(r'/Users/liusendong/Desktop/Concat/all.csv', index=False,encoding="utf_8_sig")

#输出文件

def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


M_views = pd.read_csv('/Users/liusendong/Desktop/Concat/all.csv')
M_views['cpc'] = MaxMinNormalization(M_views[['cpc']])
M_views['cpm'] = MaxMinNormalization(M_views[['cpm']])
M_views.to_csv(r'/Users/liusendong/Desktop/Concat/result.csv', index=False,encoding="utf_8_sig")
 



data=pd.read_csv('/Users/liusendong/Desktop/Concat/result_train.csv')

# 查看真异常检测二维分布图
fig, ax = plt.subplots(figsize=(10,6))
ax1 = ax.scatter(data.query("is_anomaly == True").cpc, data.query("is_anomaly == True").cpm, edgecolor = 'k', color = 'r')
ax2 = ax.scatter(data.query("is_anomaly == False").cpc, data.query("is_anomaly == False").cpm, edgecolor = 'k', color = 'b')
ax.legend([ax1, ax2], ['abnormal', 'normal'])
ax.set_xlabel('cpc')
ax.set_ylabel('cpm')
ax.set_title('Real Anomaly')
plt.show()



