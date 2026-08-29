import numpy as np
from abc import ABC,abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self,x):
        pass

    @abstractmethod
    def backward(self,dout):
        pass


class Affine(Layer):
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
        self.dweight=None
        self.dbias=None
        self.inputs=None
    def forward(self,inputs):
        self.inputs =inputs
        out= np.dot(self.inputs,self.weight)+self.bias
        return out

    def backward(self,dout):
        dinputs=np.dot(dout,self.weight.T)
        self.dweight =np.dot(self.inputs.T,dout)
        self.dbias =np.sum(dout,axis=0)
        return dinputs
        
class Linear(Layer):
    def forward(self,inputs):
        return inputs

    def backward(self,dout):
        return dout      
        
class Relu(Layer):
    def __init__(self):
        self.mask =None
    def forward(self,inputs):
        self.mask =(inputs<= 0)
        out =inputs.copy()
        out[self.mask] =0
        return out
    def backward(self,dout):
        dinputs = dout.copy()
        dinputs[self.mask] =0
        return dinputs
    
class Sigmoid(Layer):
    def __init__(self):
        self.out= None

    def forward(self,inputs):
        out= 1/(1+np.exp(-inputs))
        self.out= out
        return out

    def backward(self,dout):
        dinputs= dout*(1.0-self.out)*self.out
        return dinputs
        
class Tanh(Layer):
    def __init__(self):
        self.out = None

    def forward(self,inputs):
        out =np.tanh(inputs)
        self.out =out
        return out

    def backward(self,dout):
        dinputs= dout*(1.0-self.out**2)
        return dinputs

class Identity(Layer):
    def forward(self,inputs):
        return inputs

    def backward(self,dout):
        return dout
        
class Softmax(Layer):
    def __init__(self):
        self.out = None

    def forward(self, inputs):
        input_max =np.max(inputs,axis=1,keepdims=True)
        exp_x =np.exp(inputs - input_max)
        self.out =exp_x/np.sum(exp_x,axis=1,keepdims=True)
        return self.out

    def backward(self,dout):
        dinputs = self.out * (dout - np.sum(dout * self.out, axis=1, keepdims=True))
        return dinputs


class SoftmaxCrossEntropy(Layer):
    def __init__(self):
        self.softmax =Softmax()
        self.loss =None
        self.y =None
        self.t =None

    def forward(self,inputs,t):
        self.t =t
        self.y =self.softmax.forward(inputs)
        batch_size =self.y.shape[0]
        self.loss =-np.sum(self.t*np.log(self.y+1e-7))/batch_size
        return self.loss

    def backward(self,dout=1):
        batch_size =self.t.shape[0]
        dinputs =(self.y-self.t)/batch_size
        return dinputs

class MeanSquaredError(Layer):
    def __init__(self):
        self.identity =Identity()
        self.loss =None
        self.y =None
        self.t =None

    def forward(self,inputs,t):
        self.t =t
        self.y =self.identity.forward(inputs)
        batch_size =self.y.shape[0]
        self.loss = 0.5*np.sum((self.y-self.t)**2)/batch_size
        return self.loss

    def backward(self,dout=1):
        batch_size =self.t.shape[0]
        dinputs =(self.y-self.t)/batch_size
        return dinputs

class Dropout(Layer):
    def __init__(self,dropout_ratio):
        self.dropout_ratio =dropout_ratio
        self.mask =None
        
    def forward(self,inputs,train_flag=True):
        if train_flag:
            self.mask =np.random.rand(*inputs.shape)>self.dropout_ratio
            return inputs*self.mask
        else:
            return inputs*(1.0-self.dropout_ratio)
        
    def backward(self,dout):
        pass
        return dout*self.mask

class BatchNormalization(Layer):
    def __init__(self,gamma,beta,momentum=0.9,running_mean=None,running_var=None):
        self.gamma =gamma
        self.beta =beta
        self.momentum =momentum
        self.running_mean =running_mean
        self.running_var =running_var
        self.batch_size =None
        self.inputsc =None
        self.inputsn =None
        self.std =None
        self.dgamma =None
        self.dbeta =None

    def forward(self,inputs,train_flag=True):
        if self.running_mean is None:
            _, D =inputs.shape
            self.running_mean =np.zeros(D)
            self.running_var =np.zeros(D)
        if train_flag:
            mu =inputs.mean(axis=0)
            inputsc =inputs-mu
            var =np.mean(inputsc**2,axis=0)
            std =np.sqrt(var+1e-7)
            inputsn =inputsc/std
            self.batch_size =inputs.shape[0]
            self.inputsc =inputsc
            self.inputsn =inputsn
            self.std =std
            self.running_mean =self.momentum*self.running_mean + (1-self.momentum)*mu
            self.running_var =self.momentum*self.running_var + (1-self.momentum)*var
            out =self.gamma * inputsn + self.beta
        else:
            inputsc =inputs-self.running_mean
            inputsn =inputsc/np.sqrt(self.running_var + 1e-7)
            out =self.gamma * inputsn + self.beta
        return out
    
    def backward(self,dout):
        dbeta =dout.sum(axis=0)
        dgamma =np.sum(self.inputsn*dout,axis=0)
        dinputsn =self.gamma*dout
        dinputsc =dinputsn/self.std
        dstd =-np.sum((dinputsn*self.inputsc)/(self.std**2),axis=0)
        dvar =0.5*dstd/self.std
        dinputsc +=(2.0/self.batch_size)*self.inputsc*dvar
        dmu =-np.sum(dinputsc,axis=0)
        dinputs =dinputsc + dmu/self.batch_size
        self.dgamma =dgamma
        self.dbeta =dbeta
        return dinputs
