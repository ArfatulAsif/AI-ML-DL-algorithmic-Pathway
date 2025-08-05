
---
# Model Architecture Hyperparameters
These define the structure of the network itself.

-   **Number of Hidden Layers**: The **depth** of the network (how many layers are stacked between the input and output).
-   **Number of Neurons per Layer**: The **width** of each hidden layer.
-   **Activation Functions**: The non-linear function used in the hidden layers (e.g., **ReLU**, **Tanh**, **Leaky ReLU**).
-   **Dropout Rate (dropout_rate):**  `Regularization` Used to prevent overfitting by randomly setting some neuron outputs to zero during training.



### **Number of Hidden Layers and Neurons in Each Layer: A Brief Note**

#### **Fix Number of Neurons vs. Progressive Method**

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

#### **A General Approach: Progressive Method with Dropout**

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

## For `general-purpose architectures`, the `progressive method` is often a more regularized and efficient choice, 




---
---
---


# Training Process Hyperparameters
These control how the model learns from data.

-   **Optimizer**: The algorithm used to update the network's weights (e.g., **Adam (default and safe optimized)**, **SGD (stochastic gradient decent, Takes small steps using one batch at a time)**, **RMSprop (best for time series)**).
-   **Learning Rate**: The step size used by the optimizer to update weights.
-   **Batch Size**: The number of training samples processed before the model's weights are updated.
-   **Number of Epochs**: The number of times the model cycles through the entire training dataset.




### **1. Building the Model (Progressive Method)**

We’ll define the FNN model using the **progressive method** where the number of neurons decreases progressively as we go deeper into the network. We will also include **Dropout** to prevent overfitting.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# FNN Model with Progressive Neurons (Halving neurons without specifying number of layers)
def build_fnn_progressive(initial_neurons=200, num_layers=5, activation='relu', dropout_rate=0.5, optimizer='adam', learning_rate=0.001):
    model = Sequential()

    # First layer with initial number of neurons
    model.add(Dense(initial_neurons, input_dim=10, activation=activation))  # 10 features in the input layer

    neurons = initial_neurons

    # Add progressively smaller hidden layers (halve the neurons in each layer)
    for _ in range(1, num_layers):  # Starting from 1 because the first layer is already added
        neurons = max(neurons // 2, 1)  # Halve the number of neurons each time (ensure it's at least 1)
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))  # Add Dropout layer to prevent overfitting

    # Output layer (for classification task)
    model.add(Dense(3, activation='softmax'))  # 3 classes for Iris dataset

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

---

### **2. Hyperparameter Tuning with GridSearchCV, RandomizedSearchCV, and Optuna**


---

#### **2.1. Hyperparameter Tuning with GridSearchCV**

GridSearchCV exhaustively searches over a specified parameter grid and evaluates all combinations.

```python
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Wrapper function for Keras model to use with GridSearchCV
def create_model(num_layers=5, initial_neurons=200, activation='relu', dropout_rate=0.5, learning_rate=0.001):
    model = build_fnn_progressive(initial_neurons, num_layers, activation, dropout_rate, learning_rate=learning_rate)
    return model

# Wrap the model for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define parameter grid for GridSearchCV
param_grid = {
    'num_layers': [3, 4, 5],  # Number of hidden layers
    'initial_neurons': [100, 150, 200],  # Initial number of neurons
    'activation': ['relu', 'tanh'],  # Activation function
    'dropout_rate': [0.2, 0.5],  # Dropout rate
    'learning_rate': [0.001, 0.01],  # Learning rate
    'batch_size': [16, 32],  # Batch size
    'epochs': [10, 20]  # Number of epochs
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Fit GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

---

#### **2.2. Hyperparameter Tuning with RandomizedSearchCV**

RandomizedSearchCV randomly samples a fixed number of hyperparameter combinations from the specified grid, and is more efficient than GridSearchCV when dealing with a large hyperparameter space.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter distributions for RandomizedSearchCV
param_dist = {
    'num_layers': randint(3, 6),  # Random number of hidden layers
    'initial_neurons': randint(100, 300),  # Random initial number of neurons
    'activation': ['relu', 'tanh'],  # Randomly sample activation functions
    'dropout_rate': uniform(0, 0.5),  # Random dropout rate
    'learning_rate': uniform(1e-5, 1e-1),  # Random learning rate
    'batch_size': [16, 32, 64],  # Random batch size
    'epochs': randint(10, 50)  # Random number of epochs
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, n_jobs=-1, cv=3)

# Fit RandomizedSearchCV to the data
random_search.fit(X_train, y_train)

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
```

---

#### **2.3. Hyperparameter Tuning with Optuna**

Optuna is a more advanced hyperparameter optimization framework that uses algorithms like **Tree-structured Parzen Estimator (TPE)** for efficient search. It’s particularly good for finding **optimal hyperparameters** for deep learning models.

```python
import optuna
from sklearn.metrics import accuracy_score

# Define the objective function for Optuna
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 3, 5)
    initial_neurons = trial.suggest_int('initial_neurons', 100, 300)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50)
    
    # Build and train the model
    model = build_fnn_progressive(initial_neurons, num_layers, activation, dropout_rate, learning_rate=learning_rate)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy  # Optuna will try to maximize this value

# Create an Optuna study
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and the corresponding accuracy
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)
```

---

