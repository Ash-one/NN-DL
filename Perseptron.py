import numpy as np

def sign_func(x):
    # 符号函数
    if x >= 0:
        return 1
    else:
        return -1
    
class Perceptron:
    # 感知机模型
    def __init__(self,dim,lr=0.1,epoch=10,activate_func=sign_func):
        self.dim = dim
        self.weight = np.random.rand(dim)
        self.bias = np.random.rand(1)
        self.lr = lr
        self.epoch = epoch
        self.activate_func = activate_func
        self.last_loss = None
    
    def __str__(self):
        return f'weight:{self.weight} bias:{self.bias} loss:{self.last_loss}'
    
    def loss(self,X,Y):
        # 损失函数
        l = 0
        for x,y in zip(X,Y):
            u = np.dot(self.weight,x) + self.bias
            y_pred = self.activate_func(u)
            if y_pred != y:
                l += 1
        self.last_loss = l
        return l

    def train_one_step(self,x,y):
        # 训练单步
        # 梯度下降法调整权重
        u = np.dot(self.weight,x) + self.bias
        y_pred = self.activate_func(u)
        if y_pred != y:
            self.weight += np.dot(self.lr * y,x)
            self.bias += self.lr * y
        return self.weight,self.bias

    def train(self,X,Y):
        # 训练
        for i in range(self.epoch):
            for x,y in zip(X,Y):
                self.train_one_step(x,y)
                self.loss(X,Y)
            # print('epoch:',i,'weight:',self.weight,'bias:',self.bias,'loss:',self.loss(X,Y)

    def predict(self,x):
        # 预测
        u = np.dot(self.weight,x) + self.bias
        return self.activate_func(u)


if __name__ == '__main__':
    # 与运算的数据集，以-1作为false，1作为true
    X = [(-1,-1),(-1,1),(1,-1),(1,1)]
    Y_and = [-1,-1,-1,1]
    Y_or = [-1,1,1,1]
    Y_xor = [-1,1,1,-1]
    
    p_and = Perceptron(dim=2,lr=0.3,epoch=20)
    p_and.train(X,Y_and)

    p_or = Perceptron(dim=2,lr=0.3,epoch=20)
    p_or.train(X,Y_or)

    p_xor = Perceptron(dim=2,lr=0.3,epoch=20)
    p_xor.train(X,Y_xor)

    # 测试
    for x in X:
        print('AND',x,p_and.predict(x))
        print('OR',x,p_or.predict(x))
    print('异或问题：')
    for x in X:
        print('XOR',x,p_xor.predict(x))
