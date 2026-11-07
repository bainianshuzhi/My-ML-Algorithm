"""
逻辑回归
"""

import numpy as np

# X 包含两个特征：第一列是【学习时长(小时)】，第二列是【出勤率】
# 总共 8 个样本
X = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 1.0],
    [4.0, 3.0],
    [1.0, 1.0],
    [2.0, 1.0],
    [3.0, 3.0],
    [4.0, 2.0]
])

# y 是标签：1代表【通过考试】，0代表【未通过】
y = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]).reshape(-1, 1)

#以上由Deepseek友情提供

w=np.ones((2,1))
b=1.0
count=100000
def sigmoid(z): #反复、大量调用sigmoid函数是否会导致运行速度变慢呢？
    return 1/(1+np.exp(-z)) #答：不会，因为传入的是矩阵，Numpy会算出所有结果后再返回矩阵，这个过程与np.dot()其实差不多
    #似乎可以进行优化，通过return 1/(1+np.exp(-np.clip(z,-500,500)))达到修剪效果，保证数值稳定性
def logistic_regression(X,y,w,b):
    rate=0.01
    m=len(y)
    for i in range(count):
        error=(sigmoid(np.dot(X,w)+b)-y)
        d_w=np.dot(X.T,error)/m
        d_b=np.sum(error)/m
        w=w-rate*d_w
        b=b-rate*d_b
    p=sigmoid(np.dot(X,w)+b)
    print("最终概率矩阵：",p) #打印最终概率，采纳Deepseek的建议
    print("具体误差矩阵：",p-y) #按照自己的idea改进了一下，感觉这样的矩阵更直观。经查询，这个东西的名字叫作预测残差
    return w,b

w_end,b_end=logistic_regression(X,y,w,b)
print(w_end,b_end)