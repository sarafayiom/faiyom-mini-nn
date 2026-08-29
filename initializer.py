import numpy as np
from abc import ABC,abstractmethod


class Initializers(ABC):
    @abstractmethod
    def  initialize(self,node_num,next_node_num):
        pass

class XavierInit(Initializers):
    def initialize(self, node_num,next_node_num):
       weights =np.random.randn(node_num,next_node_num) / np.sqrt(node_num)
       return weights
    
class HeInit(Initializers):
    def initialize(self,node_num,next_node_num):
        weights =np.random.randn(node_num,next_node_num)*np.sqrt(2.0/node_num)
        return weights
    
class NormalInit(Initializers):
    def initialize(self,node_num,next_node_num):
        weights =0.01*np.random.randn(node_num,next_node_num)
        return weights
