"""
多次手搓：多元线性回归的梯度下降（含特征缩放）
"""

import numpy as np

# 特征矩阵 X：第一列是房屋面积（极大），第二列是房间数（极小）
X = np.array([
    [1000.0, 1.0],
    [1500.0, 2.0],
    [2000.0, 3.0],
    [2500.0, 4.0],
    [3000.0, 5.0]
])

# 真实规律设定为：价格 = 0.1 * 面积 + 50 * 房间数 + 10
# 所以真实的 w = [0.1, 50.0]，真实的 b = 10.0
y = np.array([160.0, 260.0, 360.0, 460.0, 560.0]).reshape(-1, 1)

#以上由Deepseek提供

w=np.ones((2,1))
b=1.0
def linear_regression(X,y,w,b):
    means=np.mean(X,axis=0)
    stand=np.std(X,axis=0)
    X=(X-means)/stand
    m=len(y)
    rate=0.01
    count=100000
    for i in range(count):
        f=np.dot(X, w)+b#shape=(5,1)
        error=f-y
        d_w=np.dot(X.T,error)/m#shape=(2,5)@(5,1)=(2,1)
        #print(d_w),shape=(2,1)
        d_b=np.sum(error)/m
        w=w-rate*d_w
        #print(w),shape=(2,1)
        b=b-rate*d_b
    w=w/(stand.reshape(-1,1))   #我吃柠檬！居然没发现w/stand会扩大矩阵！
    #print(w)
    b=b-np.dot(w.reshape((1,-1)),means) #我去，发现坑点，means形状居然是（2,），因为这是一维数组,不是列向量
                                        # 写的时候恰巧写对了，写后面算法时才发现这个坑点
    return w,b

w_end,b_end=linear_regression(X,y,w,b)
print(w_end,b_end)
