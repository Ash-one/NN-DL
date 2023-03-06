import numpy as np

def step_func(x):
    # 阶跃函数
    if x > 0:
        return 1
    else:
        return 0


class MP:
    # M-P模型类
    def __init__(self,w,theta,activate_func=step_func):
        self.theta = theta  # 阈值
        self.w = w          # 权值
        self.activate_func = activate_func  # 激活函数
    
    def forward(self,x):
        self.x = x
        self.u = np.dot(self.w,x)
        return self.activate_func(self.u - self.theta)
    

# 设计与或非的MP模型参数
mp_and = MP(w=np.array([1,1]),theta=1.5)
mp_or = MP(w=np.array([1,1]),theta=0.5)
mp_not = MP(w=np.array([-2]),theta=-1)

for x in [(0,0),(0,1),(1,0),(1,1)]:
    print('AND',x,mp_and.forward(x))
    print('OR',x,mp_or.forward(x))

for x in [0,1]:    
    print('NOT',x,mp_not.forward(x))
