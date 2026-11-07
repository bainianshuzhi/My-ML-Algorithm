import numpy as np

class MLP_two_layers:
    def __init__(self,input_size,hidden_size,output_size): #input_size是特征的数量，hidden_size是第一层神经元的数量
        self.W1=np.random.randn(input_size,hidden_size)
        self.W2=np.random.randn(hidden_size,output_size)
        self.b1=np.ones((1,hidden_size)) #注意此处与吴恩达的写法不同，b写为行向量，而非列向量
        self.b2=np.ones((1,output_size))

    def sigmoid(self,z):
        return 1/(1+np.exp(-np.clip(z,-500,500)))

    def ReLU(self,z):
        return np.maximum(0,z) #不是"maxinum"，这与number无关，而是来自于拉丁语maximus，然后用-um表示最高级

    def forward_propagation_two_layers(self,X):
        z1=np.matmul(X,self.W1)+self.b1 #哦！幸亏AI提醒，没想到b1和b2前忘记加self.了
        a1=self.ReLU(z1)
        z2=np.matmul(a1,self.W2)+self.b2
        a2=self.sigmoid(z2)
        return a2

if __name__=="__main__":
    X=np.random.randn(3,4) #X=(sample,input_size)
    tl=MLP_two_layers(4,5,2)
    print("tl.W1=",tl.W1)
    print("tl.b1=",tl.b1)
    print("tl.W2=",tl.W2)
    print("tl.b2=",tl.b2)
    a_out=tl.forward_propagation_two_layers(X)
    print("a_out=",a_out)
