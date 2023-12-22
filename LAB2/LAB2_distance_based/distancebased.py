import pandas as pd
import numpy as np
import math
import csv
import copy

# python distancebased.py
# distance based

data = pd.read_csv('/Users/liusendong/Desktop/LAB2/DataSet/result_train.csv')
x_cpc = data['cpc']
y_cpm = data['cpm']
data_anomaly =data['is_anomaly']
anomaly = [0]*953
for i in range(0,953):
    if data_anomaly[i] == "TRUE":
        anomaly[i] = 'TRUE'
    else:
        anomaly[i] = 'FALSE'

r_list = [0.05,0.1,0.2,0.25,0.3,0,37,0,4,0.45,0.55,0.67,0.74,0.82,0.91,0.95,0.99]
pai_list = [0.47,0.55,0.68,0.75,0.89,0.94]
correct = 0

#利用以下算法，训练并求得求得合适的r和pai
for r in rarray:
    for pai in paiarray:
        result = train_distance(r,pai,x,y)
        accuary = acctrain(result,data)
        if accuary>correct:
            correct = accuary
print(str(accuary)+' '+ str(r) +' ' + str(correct))

def train_distance(r,pai,x,y,data):
    result = [0]*len(data)
    lens =len(data)
    for i in range(0,lens):
        count = 0 #每一次循环开始重新计数
        for j in range(0,lens):
            if j-i != 0:
                #如果不是同一点，则循环计算不同点的距离
                dis = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                #若距离小于r，则计数加一
                if dis <= r:
                    count = count+1
        #若计数不到总样本量的pai的比例，则将该点记做异常点，否则为正常点，用bool变量表示
        if count < pai*lens:
            result[i] = 'TRUE'
        else:
            result[i] = 'FALSE'
    return result
    
#计算结果正确性
def acctrain(result,data):
    count = 0.00
    length = len(data)
    for i in range(0,length):
        if result[i] == data[i]:
            count = count+1
    return count/length
    
output = train_distance(0.78,0.8,x_cpc,y_cpm,data)
print("accuary =")
print(acctrain(output,anomaly))
