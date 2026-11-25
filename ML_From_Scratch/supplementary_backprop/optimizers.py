import numpy as np

class SGD:
    def __init__(self,params,grads,theta=0.01): #params和grads为字典
        self.params=params
        self.grads=grads
        self.theta=theta #学习率

    def update(self,params,grads):
        self.params = params
        self.grads = grads
        for key in self.params.keys():
            self.params[key]=self.params[key]-self.theta*self.grads[key]
        return self.params

class Momentum:
    def __init__(self,params,grads,theta=0.01,alpha=0.9):
        self.params=params
        self.grads=grads
        self.theta=theta
        self.alpha=alpha
        self.v={}
        for key in self.params:
            self.v[key]=np.zeros_like(self.grads[key])

    def update(self,params,grads):
        self.params = params
        self.grads = grads
        for key in self.params.keys():
            self.v[key]=self.alpha*self.v[key]-self.theta*self.grads[key]
            self.params[key]=self.params[key]+self.v[key]
        return self.params

class Adagrad:
    def __init__(self, params, grads, theta=0.01):
        self.params=params
        self.grads=grads
        self.theta=theta
        self.h={}
        for key in params.keys():
            self.h[key]=np.zeros_like(self.grads[key])

    def update(self,params,grads):
        self.params = params
        self.grads = grads
        for key in self.params.keys():
            self.h[key]=self.h[key]+self.grads[key]*self.grads[key]
            self.params[key]=self.params[key]-self.theta*self.grads[key]/(np.sqrt(self.h[key])+1e-8)
        return self.params

class Adam: #鱼书里没有细讲，故从Deepseek询问相关数学公式
    def __init__(self,params,grads,theta=0.001,beta1=0.9,beta2=0.999):
        self.params=params
        self.grads=grads
        self.theta=theta
        self.beta1=beta1
        self.beta2=beta2
        self.v={}
        self.m={}
        self.count=0
        for key in self.params:
            self.v[key]=np.zeros_like(self.grads[key])
        for key in params.keys():
            self.m[key]=np.zeros_like(self.grads[key])

    def update(self,params,grads):
        self.params = params
        self.grads = grads
        self.count+=1
        for key in self.params.keys():
            self.m[key]=self.beta1*self.m[key]+(1-self.beta1)*self.grads[key]
            self.v[key]=self.beta2*self.v[key]+(1-self.beta2)*self.grads[key]*self.grads[key]
            m_hat=self.m[key]/(1-self.beta1**self.count)
            v_hat=self.v[key]/(1-self.beta2**self.count)
            self.params[key]=self.params[key]-self.theta*m_hat/(np.sqrt(v_hat)+1e-8)
        return self.params