import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self,params,grads):
     pass

class SGD(Optimizer):
    def __init__(self,learning_rate=0.01):
        self.learning_rate =learning_rate

    def update(self,params,grads):
        for key in params.keys():
            params[key] -=self.learning_rate*grads[key]


class Momentum(Optimizer):
    def __init__(self,learning_rate=0.01,momentum=0.9):
        self.learning_rate =learning_rate
        self.momentum =momentum
        self.velocity =None

    def update(self,params,grads):
        if self.velocity is None:
            self.velocity ={}
            for key,val in params.items():
                self.velocity[key] =np.zeros_like(val)
        for key in params.keys():
            self.velocity[key] =self.momentum*self.velocity[key]-self.learning_rate*grads[key]
            params[key] +=self.velocity[key]


class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate =learning_rate
        self.h =None
    def update(self,params,grads):
        if self.h is None:
            self.h ={}
            for key,val in params.items():
                self.h[key] =np.zeros_like(val)
        for key in params.keys():
            self.h[key] +=grads[key]*grads[key]
            params[key] -=self.learning_rate*grads[key]/(np.sqrt(self.h[key])+1e-7)


class Adam(Optimizer):
    def __init__(self,learning_rate=0.001,beta1=0.9,beta2=0.999):
        self.learning_rate =learning_rate
        self.beta1 =beta1
        self.beta2 =beta2
        self.iter =0
        self.m =None
        self.v =None

    def update(self,params,grads):
        if self.m is None:
            self.m, self.v = {},{}
            for key, val in params.items():
                self.m[key] =np.zeros_like(val)
                self.v[key] =np.zeros_like(val)
        self.iter +=1
        lr_t = self.learning_rate*np.sqrt(1.0-self.beta2**self.iter)/(1.0-self.beta1**self.iter)
        for key in params.keys():
            self.m[key] +=(1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] +=(1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -=lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)