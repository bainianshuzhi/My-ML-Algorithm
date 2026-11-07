"""
多次手搓：正则化线性回归（含特征缩放）
"""

import numpy as np

# 特征 X：第一列是【面积（极大）】，第二列是【房间数（极小）】
X = np.array([
    [1000.0, 2.0],
    [1200.0, 3.0],
    [1500.0, 3.0],
    [1800.0, 4.0],
    [2000.0, 4.0],
    [2500.0, 5.0],
    [800.0, 1.0],
    [3000.0, 6.0],
    [500.0, 10.0],  # 🔥 离群点：面积很小，但房间数奇多
    [3500.0, 1.0]   # 🔥 离群点：面积巨大，但房间数极少
])

# 真实规律我们设定为：价格 = 0.2 * 面积 + 30 * 房间数 + 50
# 所以真实的 w = [0.2, 30.0]，真实的 b = 50.0
y = np.array([450.0, 530.0, 590.0, 680.0, 770.0, 900.0, 290.0, 1030.0, 450.0, 730.0]).reshape(-1, 1)

#以上样本由Deepseek提供

w=np.ones((2,1))
b=1.0
def regularized_linear_regression(X,y,w,b):
    rate=0.1
    lambda_=1.0
    count=1000000
    m=len(y)
    means=np.mean(X,axis=0)
    stand=np.std(X,axis=0)
    X=(X-means)/stand
    for i in range(count):
        error=np.dot(X,w)+b-y
        d_w=(np.dot(X.T,error)+lambda_*w)/m
        d_b=np.sum(error)/m
        w=w-rate*d_w
        b=b-rate*d_b
    w=w/stand.reshape(-1,1) #这一步怎么都没想明白，对照之前的算法，才发现原来我把w的初始shape设为（10,1），并且当作（10,1）来思考了
    b=b-np.dot(w.T,means.reshape(-1,1)) #如前文所述，我在此处发现means等的坑点
    b=np.sum(b) #提纯为标量
    return w,b

w_end,b_end=regularized_linear_regression(X,y,w,b)
print("w=",w_end,"\nb=",b_end)

"""
lambda_=0.0:
w= [[ 0.20851085]
 [32.70827258]] 
b= 143.28841631999165

lambda_=1.0:
w= [[ 0.18762846]
 [28.47114382]] 
b= 196.98387603706578

lambda_=10.0:
w= [[ 0.09925552]
 [12.71790329]] 
b= 415.72535150450426
"""