from .layers import *
from .optimizer import *
from .initializer import *
from collections import OrderedDict
from enum import Enum
import numpy as np

class ActivationType(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LINEAR = "linear"
    Identity="identity"

class LossType(Enum):
    SOFTMAX_CROSS_ENTROPY = "softmax_cross_entropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"

class OptimizerType(Enum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    ADAGRAD = "adagrad"
    ADAM = "adam"

class WeightInitType(Enum):
    XAVIER = "xavier"
    HE = "he"
    NORMAL = "normal"

class LayerType(Enum):
    AFFINE = "affine"
    ACTIVATION = "activation"
    BATCH_NORM = "batch_norm"
    DROPOUT = "dropout"


class NeuralNetwork:
    def __init__(self,input_size,hidden_layers_order,
                 loss_type=LossType.SOFTMAX_CROSS_ENTROPY,
                 optimizer_type=OptimizerType.SGD,
                 init_type=WeightInitType.XAVIER,
                 learningrate=0.01):
        self.params ={}
        self.layers =OrderedDict()

        activation_map = {
            ActivationType.RELU: Relu,
            ActivationType.SIGMOID: Sigmoid,
            ActivationType.TANH: Tanh,
            ActivationType.LINEAR: Linear,
            ActivationType.Identity: Identity,
        }

        loss_map = {
            LossType.SOFTMAX_CROSS_ENTROPY: SoftmaxCrossEntropy,
            LossType.MEAN_SQUARED_ERROR: MeanSquaredError
        }

        optimizer_map = {
            OptimizerType.SGD: SGD,
            OptimizerType.MOMENTUM: Momentum,
            OptimizerType.ADAGRAD: AdaGrad,
            OptimizerType.ADAM: Adam
        }

        initializer_map = {
            WeightInitType.XAVIER: XavierInit,
            WeightInitType.HE: HeInit,
            WeightInitType.NORMAL: NormalInit
        }

        initializer =initializer_map[init_type]()
        affine_count=1
        activation_count=1
        batchnorm_count=1
        dropout_count=1
        current_in_size = input_size 
        
        for layer_num, layers in enumerate(hidden_layers_order, start=1):
            layer_type = layers[0]
            if layer_type == LayerType.AFFINE:
                out_size = layers[1]
                weight = initializer.initialize(current_in_size, out_size) 
                bias = np.zeros(out_size)
                self.params[f"W{affine_count}"] = weight
                self.params[f"b{affine_count}"] = bias
                self.layers[f"Affine{affine_count}"] = Affine(weight, bias)
                current_in_size = out_size 
                affine_count += 1


            elif layer_type==LayerType.ACTIVATION:
                activation_type =layers[1]
                activation_name =activation_type.value.capitalize()
                self.layers[f"{activation_name}{activation_count}"] =activation_map[activation_type]()
                activation_count+=1

            elif layer_type==LayerType.BATCH_NORM:
                gamma =np.ones(current_in_size)
                beta =np.zeros(current_in_size)

                self.params[f"gamma{batchnorm_count}"] =gamma
                self.params[f"beta{batchnorm_count}"] =beta

                self.layers[f"BatchNorm{batchnorm_count}"] =BatchNormalization(gamma,beta)
                batchnorm_count +=1


   
            elif layer_type==LayerType.DROPOUT:
                ratio =layers[1]
                self.layers[f"Dropout{dropout_count}"] =Dropout(ratio)
                dropout_count+=1


        self.lastLayer =loss_map[loss_type]()
        self.optimizer =optimizer_map[optimizer_type](learningrate)

  
    def predict(self,x,train_flag=False):
        for layer in self.layers.values():
            if isinstance(layer,(Dropout,BatchNormalization)):
                x = layer.forward(x,train_flag)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flag=True)
        return self.lastLayer.forward(y,t)

    def gradient(self, x, t):
        self.loss(x, t)
        dout = self.lastLayer.backward(1)
        grads = {}
        for name, layer in reversed(self.layers.items()):
            dout = layer.backward(dout)

            if isinstance(layer, Affine):
               layer_num = name.replace("Affine", "")
               grads[f"W{layer_num}"] = layer.dweight
               grads[f"b{layer_num}"] = layer.dbias

            elif isinstance(layer, BatchNormalization):
                layer_num = name.replace("BatchNorm", "")
                grads[f"gamma{layer_num}"] = layer.dgamma
                grads[f"beta{layer_num}"] = layer.dbeta

        return grads

    def update(self,grads):
        self.optimizer.update(self.params,grads)
    
    def evaluate(self,x,t):
        y = self.predict(x,train_flag=False)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.mean(y == t)
        return accuracy

    def show_layers(self):
        for name in self.layers.keys():
            print(name)
        print(self.lastLayer.__class__.__name__)
