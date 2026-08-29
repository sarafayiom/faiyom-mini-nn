import random
import numpy as np

from neural_network import ActivationType, LayerType, NeuralNetwork
class Trainer:
    def __init__(self,network,x_train,t_train,x_test,t_test,iters_num=20,
                 batch_size=100,optimizer=None):
        self.network =network
        self.x_train =x_train
        self.t_train =t_train
        self.x_test =x_test
        self.t_test =t_test
        self.iters_num =iters_num
        self.batch_size =batch_size
        self.optimizer =optimizer
        self.loss_history =[]
        self.train_acc_history =[]
        self.test_acc_history =[]

    def train_step(self):
        data_size =self.x_train.shape[0]
        mask =np.random.choice(data_size, self.batch_size)
        x_batch =self.x_train[mask]
        t_batch =self.t_train[mask]
        grads =self.network.gradient(x_batch,t_batch)
        self.optimizer.update(self.network.params, grads)
        loss =self.network.loss(x_batch,t_batch)
        self.loss_history.append(loss)

    def fit(self):
        steps = max(self.x_train.shape[0] // self.batch_size,1)
        for epoch in range(self.iters_num):
            for step in range(steps):
                self.train_step()
            train_acc = self.accuracy(self.x_train, self.t_train)
            test_acc = self.accuracy(self.x_test, self.t_test)
            self.train_acc_history.append(train_acc)
            self.test_acc_history.append(test_acc)

    def accuracy(self, x, t):
        y = self.network.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.mean(y == t)
    
    def show_results(self):
        print(f"Initial Accuracy (Train): {self.train_acc_history[0]:.4f}")
        print(f"Initial Accuracy (Test) : {self.test_acc_history[0]:.4f}")
        print(f"Final Accuracy (Train)  : {self.train_acc_history[-1]:.4f}")
        print(f"Final Accuracy (Test)   : {self.test_acc_history[-1]:.4f}")


class HyperparameterTuning:
    def __init__(self,x_train,t_train,x_val,t_val):
        self.x_train =x_train
        self.t_train =t_train
        self.x_val =x_val
        self.t_val =t_val
        self.best_params =None
        self.best_score =-float('inf')

    def run_random_search(self,learningrates,batch_sizes,iters_num_list,hidden_sizes,layers_count_list,n_iter=10):
        for i in range(n_iter):
            learningrate =random.choice(learningrates)
            batch =random.choice(batch_sizes)
            iters_num =random.choice(iters_num_list)
            hidden_size =random.choice(hidden_sizes)
            num_layers =random.choice(layers_count_list)

            hidden_layers_order =[]
            for j in range(num_layers):
                hidden_layers_order.append((LayerType.AFFINE,hidden_size))
                hidden_layers_order.append((LayerType.ACTIVATION,ActivationType.RELU))
            hidden_layers_order.append((LayerType.AFFINE,3))

            model =NeuralNetwork(
                input_size=self.x_train.shape[1],
                hidden_layers_order=hidden_layers_order,
                learningrate=learningrate
            )

            trainer =Trainer(
                model, 
                self.x_train,self.t_train, 
                self.x_val,self.t_val,
                iters_num=iters_num, 
                batch_size=batch, 
                optimizer=model.optimizer
            )
            trainer.fit()
            score =model.evaluate(self.x_val,self.t_val)
            if score >self.best_score:
                self.best_score =score
                self.best_params ={
                    'learning_rate': learningrate,
                    'batch_size': batch,
                    'iters_num':iters_num,
                    'hidden_size': hidden_size,
                    'layers_count': num_layers
                }
        
        return self.best_params,self.best_score
    def show_best_results(self):
        if self.best_params is None:
            print("No tuning results available.")
            return
            
        print(f"Best Score (Validation) : {self.best_score:.4f}")
        print(f"Best Learning Rate      : {self.best_params['learning_rate']}")
        print(f"Best Batch Size         : {self.best_params['batch_size']}")
        print(f"Best Iterations         : {self.best_params['iters_num']}")
        print(f"Best Hidden Size        : {self.best_params['hidden_size']}")
        print(f"Best Layers Count       : {self.best_params['layers_count']}")