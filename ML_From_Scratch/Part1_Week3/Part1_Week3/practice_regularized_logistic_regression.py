"""
多次手搓：正则化逻辑回归
"""

import numpy as np

# 总共 10 个样本，两个特征
X = np.array([
    [1.0, 1.0], [1.5, 1.2], [1.8, 1.8], [2.2, 2.0], # 左下簇：大概率是坏样本(0)
    [4.0, 4.0], [4.5, 4.2], [5.0, 5.0], [5.5, 4.8], # 右上簇：大概率是好样本(1)
    [1.0, 4.0], # 【离群点1】：坏样本(0)，但是特征值很大
    [5.0, 1.0]  # 【离群点2】：好样本(1)，但是特征值很小
])

y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)

#以上样本由Deepseek提供

w=np.ones((2,1))
b=1.0
def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-500,500)))
def regularized_logistic_regression(X,y,w,b):
    rate=0.1
    m=len(y)
    count=100000
    for i in range(count):
        lambda_=1.0
        f=sigmoid(np.dot(X,w)+b) #nb，一开始居然写成了f=np.dot(X,w)+b，忘记调用sigmoid了
        error=f-y
        d_w=np.dot(X.T,error)/m+lambda_*w
        d_b=np.sum(error)/m
        w=w-rate*d_w
        b=b-rate*d_b
    print("预测残差向量：\n",sigmoid(np.dot(X,w)+b)-y)
    return w,b

w_end,b_end=regularized_logistic_regression(X,y,w,b)
print("w=",w_end,"\nb=",b_end)

"""
lambda_=0.0:
w= [[5.02981043]
 [0.92894807]] 
b= -18.058875235333325

lambda_=1.0:
w= [[0.4553086 ]
 [0.19833219]] 
b= -2.0018884511942296
"""