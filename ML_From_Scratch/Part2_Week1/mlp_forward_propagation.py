import numpy as np

X = [
[1.0, 2.0],
[3.0, 4.0],
[5.0, 6.0],
[7.0, 8.0]
]

W1 = [
[0.1, 0.2, 0.3],
[0.4, 0.5, 0.6]
]

b1 = [0.1, 0.1, 0.1]

W2 = [
[0.1],
[0.2],
[0.3]
]

b2 = [0.1]

#以上样本由Deepseek提供

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward_propagation(X, W, b):
    z=np.matmul(X,W)+b #对X和W1，a1形状（4,3）；对a1和W2，a2形状（4,1）
    a=sigmoid(z)
    return a

a1=forward_propagation(X,W1,b1)
a2=forward_propagation(a1,W2,b2)
print("a1=",a1)
print("a2=",a2)
