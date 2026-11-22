import numpy as np

class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None

    def forward(self,x,y):
        self.x=x
        self.y=y
        z=x*y
        return z

    def backward(self,z):
        dx=z*self.y
        dy=z*self.x
        return dx,dy

class AddLayer:
    def __init__(self):
        self.x=None
        self.y=None

    def forward(self,x,y):
        self.x=x
        self.y=y
        z=x+y
        return z

    def backward(self,z):
        dx=z
        dy=z
        return dx,dy

class ReLU:
    def __init__(self):
        self.x=None

    def forward(self,x):
        self.x=x
        bool_z=(x>0)
        z=x*bool_z
        return z

    def backward(self,z):
        bool_dx=(self.x>0)
        dx=z*bool_dx.astype(float)
        return dx

class Sigmoid:
    def __init__(self):
        self.x=None
        self.y=None

    def forward(self,x):
        self.x=x
        y=1/(1+np.exp(-np.clip(x,-500,500)))
        self.y=y
        return y

    def backward(self,z):
        dx=z*self.y*(1-self.y)
        return dx

class Affine:
    def __init__(self):
        self.X=None
        self.W=None
        self.b=None

    def forward(self,X,W,b):
        self.X=X
        self.W=W
        self.b=b
        Y=np.matmul(X,W)+b
        return Y

    def backward(self,Z):
        db=np.sum(Z,axis=0)
        dX=np.matmul(Z,self.W.T)
        dW=np.matmul(self.X.T,Z)
        return dX,dW,db

class SoftmaxWithLoss:
    def __init__(self):
        self.A=None
        self.Y=None
        self.t=None

    def forward(self,A,t):
        self.A=A
        self.t=t
        A_max=np.max(A,axis=1).reshape(-1,1)
        A=A-A_max
        exp_A=np.exp(A)
        exp_A_sum=np.sum(exp_A,axis=1).reshape(-1,1)+(1e-8)
        Y=exp_A/exp_A_sum
        self.Y=Y
        L=np.sum(-t*np.log(Y+1e-7))/A.shape[0]
        return L

    def backward(self,Z):
        dA=Z*(self.Y-self.t)/self.Y.shape[0]
        return dA

