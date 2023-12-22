import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#类定义
class CyrusDecisionTree(object):
    X = None
    Y = None

    def __init__(self, criterion="C4.5"):
        self.criterion = criterion
        self.tree_net = None

    # 1、计算信息熵的函数
    @classmethod
    def cal_entropy(class_obj, y):
        y = np.array(y).reshape(-1)
        counts = np.array(pd.Series(y).value_counts())
        return -((counts / y.shape[0]) * np.log2(counts / y.shape[0])).sum()

    # 2、计算条件熵的函数
    @classmethod
    def cal_conditional_entropy(class_obj, x, y):
        """
        计算在条件x下y的信息熵
        """
        x = np.array(pd.Series(x).sort_values()).reshape(-1)
        y = np.array(y).reshape(-1)[list(pd.Series(x).argsort())]
        split = []
        entropy = []
        for i in range(x.shape[0] - 1):
            split.append(0.5 * (x[i] + x[i + 1]))
            entropy.append((i + 1) / x.shape[0] * class_obj.cal_entropy(y[:i + 1]) + (
                        1 - (i + 1) / x.shape[0]) * class_obj.cal_entropy(y[i + 1:]))
        return (np.array(entropy), np.array(split))

    # 3、计算信息增益的函数
    @classmethod
    def cal_entropy_gain(class_obj, x, y):
        """
        计算在条件x下y的信息增益
        """
        entropy, split = class_obj.cal_conditional_entropy(x, y)
        entropy_gain = class_obj.cal_entropy(y) - entropy
        return entropy_gain.max(), split[entropy_gain.argmax()]

    # 4、计算熵增益率的函数
    @classmethod
    def cal_entropy_gain_ratio(class_obj, x, y):
        """
        计算在条件x下y的信息增益率
        """
        entropy_gain, split = class_obj.cal_entropy_gain(x, y)
        entropy_condition = class_obj.cal_entropy(y) - entropy_gain
        return entropy_gain / entropy_condition, split

    # 5、Gini系数计算函数
    @classmethod
    def cal_gini(class_obj, y):
        y = np.array(y).reshape(-1)
        counts = np.array(pd.Series(y).value_counts())
        return 1 - (((counts / y.shape[0]) ** 2).sum())

    # 6、Gini系数增益计算
    @classmethod
    def cal_gini_gain(class_obj, x, y):
        """
        计算在条件x下y的Gini系数增益
        """
        x = np.array(pd.Series(x).sort_values()).reshape(-1)
        y = np.array(y).reshape(-1)[list(pd.Series(x).argsort())]
        split = []
        gini = []
        for i in range(x.shape[0] - 1):
            split.append(0.5 * (x[i] + x[i + 1]))
            gini.append(
                (i + 1) / x.shape[0] * class_obj.cal_gini(y[:i + 1]) + (1 - (i + 1) / x.shape[0]) * class_obj.cal_gini(
                    y[i + 1:]))
        gini_gain = class_obj.cal_gini(y) - np.array(gini)
        split = np.array(split)
        return gini_gain.max(), split[gini_gain.argmax()]

    # tree构建递归函数
    def tree(self, x, y, net):
        if pd.Series(y).value_counts().shape[0] == 1:
            net.append(y[0])
        else:
            x_entropy = []
            x_split = []
            for i in range(x.shape[1]):
                if self.criterion == "C4.5":
                    entropy, split = self.cal_entropy_gain_ratio(x[:, i], y)
                else:
                    entropy, split = self.cal_gini_gain(x[:, i], y)
                x_entropy.append(entropy)
                x_split.append(split)
            rank = np.array(x_entropy).argmax()
            split = x_split[rank]
            net.append(rank)
            net.append(split)
            net.append([])
            net.append([])
            x_1 = []
            x_2 = []
            for i in range(x.shape[0]):
                if x[i, rank] > split:
                    x_1.append(i)
                else:
                    x_2.append(i)
            x1 = x[x_1, :]
            y1 = y[x_1]
            x2 = x[x_2, :]
            y2 = y[x_2]
            return self.tree(x1, y1, net[2]), self.tree(x2, y2, net[3])

    def predict_tree(self, x, net):
        x = np.array(x).reshape(-1)
        if len(net) == 1:
            return net
        else:
            if x[net[0]] >= net[1]:
                return self.predict_tree(x, net[2])
            else:
                return self.predict_tree(x, net[3])

    # 模型训练函数
    def fit(self, x, y):
        self.X = np.array(x)
        self.Y = np.array(y).reshape(-1)
        self.tree_net = []
        self.tree(self.X, self.Y, self.tree_net)

    # 模型预测函数
    def predict(self, x):
        x = np.array(x)
        pre_y = []
        for i in range(x.shape[0]):
            pre_y.append(self.predict_tree(x[i, :], self.tree_net))
        return np.array(pre_y)

#

#修改文件路径
df = pd.read_csv(r'D:\DTI\result.csv')
n=len(df)

x = np.array(df.iloc[:n, 1:-1])
df['is_anomaly'] = [1 if df['is_anomaly'][i] == True else 0 for i in range(len(df))]
y = np.array(df.iloc[:n, -1])
print(x)
print(y)
model = CyrusDecisionTree(criterion="gini")
model.fit(x, y)
y_pre = model.predict(x)
print('预测标签', y_pre)
pd.DataFrame(y_pre,columns=['预测结果']).to_csv('D:\DTI\预测结果.csv',index=0)

##异常检测结果绘图
plt.scatter(x[:,0],x[:,1],c=y_pre)
plt.xlabel('cpc')
plt.ylabel('cpm')
plt.show()