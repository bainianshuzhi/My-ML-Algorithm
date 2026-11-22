import numpy as np
from layers import MulLayer, AddLayer, Affine, SoftmaxWithLoss, Sigmoid, ReLU
np.random.seed(6)

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        #x=(batch_size,input_size), w1=(input_size,hidden_size), w2=(hidden_size,output_size), b1=(1,hidden_size), b2=(1,output_size)
        self.affine1=Affine()
        self.affine2=Affine()
        self.relu1=ReLU()
        self.loss_layer=SoftmaxWithLoss()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.w1=np.random.normal(0,1,(input_size,hidden_size))*0.01
        self.w2=np.random.normal(0,1,(hidden_size,output_size))*0.01
        self.b1=np.random.normal(0, 1, (1,hidden_size))*0.01
        self.b2=np.random.normal(0, 1, (1,output_size))*0.01

    def predict(self,x):
        a1=self.affine1.forward(x,self.w1,self.b1)
        a_re=self.relu1.forward(a1)
        a2=self.affine2.forward(a_re,self.w2,self.b2)
        return a2

    def forward(self,x,t):
        y=self.predict(x)
        loss=self.loss_layer.forward(y,t)
        return loss

    def backward(self,dout=1):
        dout=self.loss_layer.backward(dout)
        dx2,dw2,db2=self.affine2.backward(dout)
        dx1=self.relu1.backward(dx2)
        _,dw1,db1=self.affine1.backward(dx1)
        return dw1,db1,dw2,db2


net=TwoLayerNet(2,3,4)
net_try=TwoLayerNet(2,3,4) #由于Deepseek指出，net会在数值微分后被污染，于是我想到可以用net_try作为数值微分的专用实例
x=np.random.normal(10,6.66,(20,2))
right=np.random.randint(0,4,size=20)
t=np.zeros((20,4))
t[np.arange(20),right]=1.0
loss=net.forward(x,t)
print("loss=",loss)


def f(w_temp):
    net_try.w1=w_temp
    net_try.w2=net.w2 #同步net和net_try的参数
    net_try.b1=net.b1
    net_try.b2=net.b2
    f_loss=net_try.forward(x,t)
    return f_loss

def numerical_gradient(w_target):
    h=1e-4
    grad=np.zeros_like(w_target)
    for i in range(w_target.shape[0]):
        for j in range(w_target.shape[1]):
            tmp=w_target[i][j]
            w_target[i][j]=tmp+h
            loss1=f(w_target)
            w_target[i][j]=tmp-h
            loss2=f(w_target)
            w_target[i][j]=tmp
            grad[i][j]=(loss1-loss2)/(2*h)
    return grad

back_dw1,_,_,_=net.backward()
nume_dw1=numerical_gradient(net.w1)
error=np.mean(np.abs(back_dw1-nume_dw1)) #验证反向传播的准确性
print("error=",error)