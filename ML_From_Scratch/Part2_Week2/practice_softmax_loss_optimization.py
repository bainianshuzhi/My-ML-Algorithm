import numpy as np

# X: (m=4, n=3)
X_sample = np.array([
    [1.0, 2.0, -1.0],
    [0.5, -1.5, 2.0],
    [-1.0, 0.0, 1.0],
    [2.0, 1.0, -0.5]
])

# Y: (m=4, k=3) one-hot 编码
Y_sample = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# 初始化参数
np.random.seed(42) # 保证可以复现
W = np.random.randn(3, 3) * 0.01
b = np.zeros((3,))

#以上由Deepseek提供

X=X_sample
Y=Y_sample

print("W=\n",W)

"""
原始代码：
def softmax(z):
    z_max=np.max(z,axis=1)
    z_max=z_max.reshape(-1,1)
    z_exp=np.exp(z-z_max)
    #z_exp不需要reshape!
    z_sum=np.sum(z_exp,axis=1)
    a=z_exp/(z_sum.reshape(-1,1))
    return a

def softmax_loss(X,Y,W,b):
    z=np.matmul(X,W)+b #z=(4,3)
    a=softmax(z)
    m=X.shape[0]
    loss=-(np.sum(Y*np.log(a)))/m
    return loss
"""

def softmax_loss(X,Y,W,b):
    z=np.matmul(X,W)+b
    m=X.shape[0]
    z_max=np.max(z,axis=1).reshape(-1,1)
    z_shifted=z-z_max
    z_exp_sum=np.sum(np.exp(z_shifted),axis=1).reshape(-1,1)
    loss=-(np.sum(Y*(z_shifted-np.log(z_exp_sum))))/m
    return loss

loss=softmax_loss(X,Y,W,b)
print("loss=\n",loss)
