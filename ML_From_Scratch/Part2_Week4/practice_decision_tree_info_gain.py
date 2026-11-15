import numpy as np

# 训练集 (16个样本)
X_train = np.array([
    [1.0, 1.0], [2.0, 2.0], [4.0, 1.5], [6.0, 1.0],
    [2.0, 4.0], [4.0, 3.0], [1.0, 4.5], [5.0, 2.0], # 以上 X0<=5 或 X1<=2.5，标签为 0
    [6.0, 3.0], [8.0, 4.0], [7.0, 5.0], [9.0, 3.0],
    [6.5, 2.8], [10.0, 4.5], [8.5, 3.5], [7.5, 2.7] # 以上 X0>5 且 X1>2.5，标签为 1
])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# 测试集 (4个样本)
X_test = np.array([
    [1.0, 5.0],  # 标签 0
    [8.0, 1.0],  # 标签 0
    [7.0, 4.0],  # 标签 1
    [6.0, 3.5]   # 标签 1
])
y_test = np.array([0, 0, 1, 1])

#以上由Deepseek提供

def entropy(p1):
    p0=1.0-p1
    if p1==1 or p1==0:
        h=0
    else:
        h=-p1*(np.log(p1)/np.log(2))-p0*(np.log(p0)/np.log(2))
    return h

def information_gain(x,y,splits,dimension):

    m_root=x.shape[0]
    m_left=0;m_right=0
    r_left=0;r_right=0;r_root=0 #统计y为真的数量

    for i in range(m_root):
        if y[i]==1:
            r_root+=1
        if x[i][dimension]<=splits:
            m_left+=1
            if y[i]==1:
                r_left+=1
        else:
            m_right+=1
            if y[i]==1:
                r_right+=1
    if m_right==0 or m_left==0:
        return 0
    p_left=r_left/m_left
    p_right=r_right/m_right
    w_left=m_left/m_root
    w_right=m_right/m_root
    gain=entropy(r_root/m_root)-(w_left*entropy(p_left)+w_right*entropy(p_right))
    return gain
