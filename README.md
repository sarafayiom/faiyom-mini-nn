# faiyom-mini-nn

A modular, lightweight Deep Learning framework built entirely **from scratch** using Python and NumPy. This framework provides a dynamic architecture that allows users to define custom hidden layers, activation functions, optimization algorithms, and weight initialization strategies to ensure stable training dynamics.

---

##  Key Features

* **Modular Layer Architecture:** Complete flexibility to stack Affine (Dense), Activation, Batch Normalization, and Dropout layers.
* **Custom Optimizers & Initializers:** Built-in support for multiple optimization techniques and weight initialization methods.
* **Dynamic Model Construction:** Flexible interface using `Enums` and `Mapping` mechanisms to parse user-defined architectures easily.
* **Automated Hyperparameter Tuning:** Integrated Random Search module to discover optimal learning rates, batch sizes, and depth configurations.

---

## 📁 Project Structure

* **`layers.py`**: Base abstract layer class and all inheriting concrete layers (Affine, Activation, Loss, Dropout, Batch Normalization).
* **`optimizer.py`**: Optimization algorithms derived from a common abstract base class (e.g., SGD, Adam).
* **`initializer.py`**: Weight initialization strategies to prevent vanishing/exploding gradients.
* **`trainer.py`**: Training loop abstraction handling batch splitting, weight updates, loss/accuracy tracking, and the automated `RandomSearch` hyperparameter tuner.
* **`neural_network.py`**: Core model wrapper that orchestrates layers, gradient computations, predictions, and architecture summary visualizers.

---

##  Usage Example

### 1. Define Model Architecture
Define hidden layers using ordered tuples specifying layer type, parameters, or activation functions:

```python
from faiyomMiniNN.neural_network import NeuralNetwork, LayerType, ActivationType

hidden_layers_order = [
    (LayerType.AFFINE, 10),
    (LayerType.ACTIVATION, ActivationType.SIGMOID),
    (LayerType.BATCH_NORM,),
    (LayerType.AFFINE, 10),
    (LayerType.ACTIVATION, ActivationType.RELU),
    (LayerType.AFFINE, 3),
    (LayerType.ACTIVATION, ActivationType.RELU),
    (LayerType.AFFINE, 3)
]

model = NeuralNetwork(
    hidden_layers_order=hidden_layers_order,
    input_size=4,
    learning_rate=0.01
)

# Display network topology
model.show_layers()
```

### 2. Train the Network
Pass the model along with training and evaluation datasets to the Trainer class:

```python
from faiyomMiniNN.trainer import Trainer
from faiyomMiniNN.optimizer import Adam

trainer = Trainer(
    model, x_train, t_train, x_test, t_test,
    iters_num=200,
    batch_size=16,
    optimizer=Adam(learning_rate=0.01)
)

print("\nTraining in progress...")
trainer.fit()
trainer.show_results()
```

### 3. Hyperparameter Tuning
Automate hyperparameter optimization using Random Search:

```python
# Run random search over specified search spaces
best_params, best_score = tuner.run_random_search(
    lrs, batches, iters_list, h_sizes, layers_counts, n_iter=5
)

# Display optimal parameters
tuner.show_best_results()
```

---

##  Sample Output (Iris Dataset Evaluation)

```text
-- Network Architecture --
Affine1
Sigmoid1
BatchNorm1
Affine2
Relu2
Affine3
Relu3
Affine4
SoftmaxCrossEntropy

Training in progress ...
Initial Accuracy (Train): 0.5333
Initial Accuracy (Test) : 0.6000
Final Accuracy (Train)  : 0.9750
Final Accuracy (Test)   : 1.0000

-- Hyperparameter Tuning --
Best Score (Validation) : 1.0000
Best Learning Rate      : 0.1
Best Batch Size         : 8
Best Iterations         : 100
Best Hidden Size        : 20
Best Layers Count       : 1

-- Final Training with Best Parameters --
Initial Accuracy (Train): 0.9667
Initial Accuracy (Test) : 1.0000
Final Accuracy (Train)  : 0.9417
Final Accuracy (Test)   : 0.9333
```

---

##  How to Run Locally

Clone the repository:
```bash
git clone https://github.com/sarafayiom/faiyom-mini-nn.git
```

Navigate to directory:
```bash
cd faiyom-mini-nn
```

Run tests/example script:
```bash
python test/main.py
```
