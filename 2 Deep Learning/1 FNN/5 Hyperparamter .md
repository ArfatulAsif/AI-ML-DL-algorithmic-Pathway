
---
# Model Architecture Hyperparameters
These define the structure of the network itself.

-   **Number of Hidden Layers**: The **depth** of the network (how many layers are stacked between the input and output).
-   **Number of Neurons per Layer**: The **width** of each hidden layer.
-   **Activation Functions**: The non-linear function used in the hidden layers (e.g., **ReLU**, **Tanh**, **Leaky ReLU**).
-   **Dropout Rate (dropout_rate):**  `Regularization` Used to prevent overfitting by randomly setting some neuron outputs to zero during training.



## **Number of Hidden Layers and Neurons in Each Layer: A Brief Note**

### **Fix Number of Neurons vs. Progressive Method**

1. **Fixed Number of Neurons**:

   * In the **fixed number of neurons** approach, each hidden layer in the network has the **same number of neurons**.
   * This method is **simple** to implement and provides **consistent network capacity** across all layers.
   * **Advantages**: It’s suitable for complex tasks where **capacity** is needed at each layer, and works well with **larger datasets**.
   * **Disadvantages**: It can lead to **overfitting** when the model has more neurons than necessary for the given task, especially with smaller datasets.

2. **Progressive Method**:

   * The **progressive method** reduces the number of neurons as we go deeper into the network. The first layer may have a large number of neurons, while subsequent layers progressively have fewer neurons.
   * This method forces the model to **compress** features as it moves deeper, preventing overfitting and improving generalization.
   * **Advantages**: It helps in **regularizing** the model, making it more **efficient** and less prone to overfitting, especially with smaller datasets.
   * **Disadvantages**: If the reduction is too aggressive, it might lead to **underfitting**, where the model doesn't have enough capacity to capture complex patterns.

---

### **A General Approach: Progressive Method with Dropout**

A good general approach combines the **progressive method** with **Dropout** for regularization:

* **Start with a large number of neurons** in the first layer (e.g., 200 neurons) to capture complex features from the data.
* Apply **Dropout** after each hidden layer to **reduce overfitting** and ensure the model generalizes well.
* **Progressively reduce the number of neurons** in each subsequent layer. For instance:

  * First layer: 200 neurons.
  * Second layer: 100 neurons.
  * Third layer: 50 neurons, and so on.
* The **number of neurons is halved** after each layer, and the final layer will contain **1 neuron** (as the number of neurons keeps halving until it reaches 1).

Since the **number of neurons** is halved, you **don’t need to specify the number of layers** explicitly — it is implicitly defined by how many times you reduce the number of neurons until you reach 1. For example, if you start with 200 neurons, after 3 layers you’ll reach 25 neurons, and after 4 layers you’ll reach 12, and so on. The number of layers is automatically determined by the halving.

---

### **Coding Example:**

#### **1. Fixed Number of Neurons**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# FNN Model with Fixed Number of Neurons
def build_fnn_fixed(num_layers=3, num_neurons=128, activation='relu', dropout_rate=0.5, optimizer='adam', learning_rate=0.001):
    model = Sequential()
    
    # Input layer
    model.add(Dense(num_neurons, input_dim=10, activation=activation))  # 10 features in the input layer

    # Hidden layers with the same number of neurons
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))
        model.add(Dropout(dropout_rate))  # Add Dropout layer to prevent overfitting
    
    # Output layer (for classification task)
    model.add(Dense(3, activation='softmax'))  # 3 classes for Iris dataset

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example model with 3 layers, 128 neurons in each layer
model_fixed = build_fnn_fixed(num_layers=3, num_neurons=128, dropout_rate=0.5)
model_fixed.summary()
```

In this fixed architecture:

* Every hidden layer has **128 neurons**.
* **Dropout** is applied after each hidden layer to reduce overfitting.
* The model has **3 hidden layers**, and you can adjust the **number of neurons** and **layers** as needed.

---

#### **2. Progressive Method (Number of Neurons Halved)**

In this approach, the number of neurons is progressively reduced by half for each subsequent layer, and the number of layers is determined by how many times you need to halve the neurons to reach 1.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# FNN Model with Progressive Neurons (halving neurons in each layer)
def build_fnn_progressive(initial_neurons=200, num_layers=5, activation='relu', dropout_rate=0.5, optimizer='adam', learning_rate=0.001):
    model = Sequential()
    
    # First layer with initial number of neurons
    model.add(Dense(initial_neurons, input_dim=10, activation=activation))  # 10 features in the input layer
    
    neurons = initial_neurons
    
    # Add progressively smaller hidden layers
    for _ in range(num_layers - 1):
        neurons = max(neurons // 2, 1)  # Halve the number of neurons (ensure it never goes below 1)
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))  # Add Dropout layer to prevent overfitting
    
    # Output layer (for classification task)
    model.add(Dense(3, activation='softmax'))  # 3 classes for Iris dataset

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example model with initial 200 neurons, 5 layers, and Dropout
model_progressive = build_fnn_progressive(initial_neurons=200, num_layers=5, dropout_rate=0.5)
model_progressive.summary()
```

In this progressive architecture:

* The first layer has **200 neurons**, and the number of neurons in subsequent layers **halves** with each layer.

  * For example: 200 → 100 → 50 → 25 → 12 neurons, and so on.
* **Dropout** is applied after each layer to prevent overfitting.
* The **number of layers** is automatically determined based on how many times neurons can be halved until they reach 1. In this case, with 200 neurons, it takes **5 layers** to reduce to 1.

### **Summary**:

* **Fixed Number of Neurons**: The same number of neurons is used in all hidden layers. This approach is simple and works well for complex tasks and larger datasets, but may risk **overfitting** without proper regularization (e.g., dropout).

* **Progressive Reduction of Neurons**: The number of neurons is progressively halved for each layer. This approach is great for **regularization**, reducing **overfitting**, and improving **generalization**, especially with smaller datasets or less complex tasks. The number of layers is determined by how many times you can halve the number of neurons until you reach 1.

---

### **When to Use Each**:

* **Fixed Number of Neurons**: Use when you need consistent capacity in each layer for complex tasks and have enough data to avoid overfitting.
* **Progressive Method**: Use when you need to regularize the model, especially with smaller datasets or when you want to reduce the model's complexity progressively.

## for **general-purpose architectures**, the **progressive method** is often a more regularized and efficient choice, 




---
# Training Process Hyperparameters
These control how the model learns from data.

-   **Optimizer**: The algorithm used to update the network's weights (e.g., **Adam (default and safe optimized)**, **SGD (stochastic gradient decent, Takes small steps using one batch at a time)**, **RMSprop (best for time series)**).
-   **Learning Rate**: The step size used by the optimizer to update weights.
-   **Batch Size**: The number of training samples processed before the model's weights are updated.
-   **Number of Epochs**: The number of times the model cycles through the entire training dataset.




```python
# First, you need to install KerasTuner
# pip install keras-tuner

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner

# 1. Load and prepare the dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# 2. Define the model-building function
# This function tells the tuner how to build the model for each trial,
# defining the search space for each hyperparameter.
def build_model(hp):
    model = keras.Sequential()
    
    # Define the input layer
    model.add(layers.Input(shape=(784,)))
    
    # Tune the number of neurons in the first hidden layer
    # Search for an integer between 64 and 512
    hp_units = hp.Int('units', min_value=64, max_value=512, step=32)
    
    # Tune the activation function
    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    
    model.add(layers.Dense(units=hp_units, activation=hp_activation))
    
    # Add a dropout layer with a tunable rate
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    model.add(layers.Dropout(rate=hp_dropout))
    
    # Add the output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Instantiate the tuner and perform the search
tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_accuracy',  # The metric to optimize
    max_trials=10,             # Number of different hyperparameter combinations to test
    directory='tuner_dir',     # Directory to store results
    project_name='fnn_tuning'
)

# Use an early stopping callback to prevent long, fruitless trials
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("Starting hyperparameter search...")
# The tuner will run 'max_trials' of model training sessions
tuner.search(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[stop_early])

# 4. Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Search complete. The optimal hyperparameters are:
- Units in hidden layer: {best_hps.get('units')}
- Activation function: {best_hps.get('activation')}
- Dropout rate: {best_hps.get('dropout'):.2f}
- Learning rate: {best_hps.get('learning_rate')}
""")

# You can now build and train the final model with these best hyperparameters
# final_model = tuner.hypermodel.build(best_hps)
# final_model.fit(...)

```
