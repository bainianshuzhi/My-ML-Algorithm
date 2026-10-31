"""
梯度下降
"""

import numpy as np
x=np.array([1.0,2.0,3,4,5])
y=np.array([300.0,500.0,700,900,1100])
w=1
b=1

def linear_regression(x,y,w,b):
    l=len(x)
    m=l
    origin=0
    rate=0.001

    while origin<10000000:
        need_use_w=0
        need_use_b=0
        for i in range(l):
            need_use_w+=(w*x[i]+b-y[i])*x[i]/m
            need_use_b+=(w*x[i]+b-y[i])/m
        temp=w
        w=w-rate*need_use_w
        b=b-rate*need_use_b
        '''if abs(w-temp)<0.0000001:
            break'''
        origin+=1
    print(w,b)


linear_regression(x,y,w,b)