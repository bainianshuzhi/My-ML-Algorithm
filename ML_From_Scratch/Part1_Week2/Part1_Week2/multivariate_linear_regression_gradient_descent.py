"""
多元线性回归的梯度下降
"""

import numpy as np
import matplotlib.pyplot as plt

# x 是一个 (4, 2) 的矩阵：4个样本，每个样本2个特征
x = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 1.0],
    [4.0, 5.0]
])

# y 是目标值（完美按照 y = 2*x1 + 3*x2 + 5 计算出来的）
# 比如第一行：2*1.0 + 3*2.0 + 5 = 13.0
y = np.array([13.0, 18.0, 14.0, 28.0])

#以上由Deepseek提供，以下Matplotlib的相关代码也由Deepseek提供

y=y.reshape(-1,1)
w=np.array([[1.0],
            [1.0]])
b=1.0
#f=w1*x1+w2*x2+b
def linear_regression(x,y,w,b):
    count=0
    rate=0.001
    park=0.000001
    loss_history=[]
    while count<100000:
        m=len(y)

        f=np.dot(x,w)+b
        error=f-y
        d_w=np.dot(x.T,error)/m
        d_b=np.sum(error)/m
        w=w-rate*d_w
        b=b-rate*d_b
        count+=1

        loss=np.sum(error**2)/(2*m)
        loss_history.append(loss)

    return w,b,loss_history

w,b,losses=linear_regression(x,y,w,b)
print("w=",w)
print("b=",b)
plt.plot(losses)
plt.xlabel("diedaicishu")
plt.ylabel("daijia")
plt.title("title")
plt.show()