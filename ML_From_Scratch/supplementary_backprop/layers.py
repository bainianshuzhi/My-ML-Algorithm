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

#后续添加，未参与到test_gradient中

class BatchNormalization:
    def __init__(self,):
        self.x=None
        self.x_hat=None
        self.mu=None
        self.var=None
        self.gamma=None
        self.beta=None
        self.y=None

    def forward(self,x):
        self.x=x
        m=self.x.shape[0]
        if self.gamma is None:
            self.gamma=np.ones(x.shape[1])
            self.beta=np.zeros(x.shape[1])
        self.mu=(np.sum(x,axis=0)/m).reshape(1,-1)
        self.var=(np.sum((x-self.mu)**2,axis=0)/m).reshape(1,-1)
        self.x_hat=(x-self.mu)/np.sqrt((self.var+1e-8))
        self.y=self.gamma*self.x_hat+self.beta
        return self.y

    def backward(self,z):
        m=self.x.shape[0]
        dgamma=np.sum(z*self.x_hat,axis=0)
        dbeta=np.sum(z,axis=0)
        dx_hat=z*self.gamma
        dx=(m*dx_hat-np.sum(dx_hat,axis=0)-self.x_hat*np.sum(dx_hat*self.x_hat,axis=0))/(m*np.sqrt(self.var+1e-8))
        return dx,dgamma,dbeta

class Dropout:
    def __init__(self):
        self.mask=None
        self.dropout_radio=None

    def forward(self,x,dropout_radio=0.5,train_flg=True):
        self.dropout_radio=dropout_radio
        if train_flg==True:
            self.mask=np.random.rand(*x.shape)>dropout_radio
            return x*self.mask/(1-self.dropout_radio) #鱼书里是传统实现，此处则采用工业标准
        else:
            return x

    def backward(self,dout):
        return dout*self.mask/(1-self.dropout_radio)