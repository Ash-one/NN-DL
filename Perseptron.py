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
        self.bias = np.random.rand(dim)
        self.lr = lr
        self.epoch = epoch
        self.activate_func = activate_func
        
    def loss(self,X,Y):
        # 损失函数
        loss = 0
        for x,y in zip(X,Y):
            u = np.dot(self.weight,x) + self.bias
            u = sum(u)
            y_pred = self.activate_func(u)
            if y_pred != y:
                loss += 1
        return loss

    def train_one_step(self,x,y):
        # 训练单步
        u = np.dot(self.weight,x) + self.bias
        u = sum(u)
        y_pred = self.activate_func(u)
        if y_pred != y:
            self.weight += np.dot(self.lr * y,x)
            self.bias += self.lr * y

    def train(self,X,Y):
        # 训练
        for i in range(self.epoch):
            for x,y in zip(X,Y):
                self.train_one_step(x,y)
            
            print('epoch:',i,'weight:',self.weight,'bias:',self.bias)
            print('loss:',self.loss(X,Y))

    def predict(self,x):
        # 预测
        u = np.dot(self.weight,x) + self.bias
        u = sum(u)
        return self.activate_func(u)



if __name__ == '__main__':
    # 与运算的数据集，以-1作为false，1作为true
    X = [(-1,-1),(-1,1),(1,-1),(1,1)]
    Y_and = [-1,-1,-1,1]
    Y_or = [-1,1,1,1]
    
    p_and = Perceptron(dim=2,lr=0.3,epoch=20)
    p_and.train(X,Y_and)

    p_or = Perceptron(dim=2,lr=0.3,epoch=20)
    p_or.train(X,Y_or)

    # 测试
    for x in X:
        print('AND',x,p_and.predict(x))
        print('OR',x,p_or.predict(x))


    

