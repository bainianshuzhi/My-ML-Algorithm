import numpy as np

# X 是 (4, 2)：4个样本，2个特征
X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 1.0],
    [4.0, 5.0]
])

# 真实的类别标签（0, 1, 2 代表三个不同的类）
# 对应第一个样本是类0，第二个类1，第三个类2，第四个类1
y_labels = np.array([0, 1, 2, 1])

# 为了计算交叉熵，我们需要把标签变成 One-Hot 编码，形状为 (4, 3)
Y = np.array([
    [1.0, 0.0, 0.0],  # 属于类别 0
    [0.0, 1.0, 0.0],  # 属于类别 1
    [0.0, 0.0, 1.0],  # 属于类别 2
    [0.0, 1.0, 0.0]   # 属于类别 1
])

# 初始化权重 W (2, 3) 和偏置 b (1, 3)
W = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])
b = np.array([[0.1, 0.1, 0.1]])

#以上由Deepseek提供

def softmax(z):
    z_max=np.max(z,axis=1) #不选择求整个z的max，是工业中防止z之间差距过大导致下溢出而形成的规范
    z_max=z_max.reshape(-1,1)
    the_exp=np.exp(z-z_max) #可以防止数值爆炸
    the_sum=np.sum(the_exp,axis=1)
    the_sum=the_sum.reshape(-1,1)
    return the_exp/the_sum

def softmax_regression_loss(X,W,b,Y):
    m=X.shape[0]
    z=np.matmul(X,W)+b #z是按列分布的，共三列，故有z1，z2，z3
    a=softmax(z)
    loss=-np.sum(Y*np.log(a))/m
    return loss

Loss=softmax_regression_loss(X,W,b,Y)
print("Loss=",Loss)