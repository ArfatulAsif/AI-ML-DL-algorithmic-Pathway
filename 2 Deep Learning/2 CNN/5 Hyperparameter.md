# Model Architecture Hyperparameters

These define the structure of the network itself.

* **Number of Convolutional Blocks**: How many repeated Conv→(Pooling) units you stack.
* **Number of Filters per Block**: The width of each convolutional layer (how many feature maps it learns).
* **Filter (Kernel) Size**: Spatial size of each convolutional kernel (e.g., 3×3, 5×5).
* **Stride**: How many pixels the filter moves at each step.
* **Padding**: Whether you pad the input (“same” vs “valid”) to control output size.
* **Activation Functions**: Non-linear transforms (e.g. **ReLU**, **Leaky ReLU**, **ELU**) applied after each convolution.
* **Pooling Type & Pool Size**: Downsampling strategy (e.g. **MaxPooling2D** with 2×2 window).
* **Dropout Rate**: Fraction of activations randomly zeroed (often after pooling or head layers) to regularize.
* **Number of Dense Layers in Head**: How many fully-connected layers you place after flattening.
* **Units per Dense Layer in Head**: Number of neurons in each of those head layers.

---

### **Number of Filters and Kernel Sizes: A Brief Note**

#### **1. Fixed Filters vs. Progressive Filters**

1. **Fixed Number of Filters**

   * Each conv layer uses the **same** number of filters (e.g. 32 filters everywhere).
   * **Advantages**: Simpler; uniform capacity per layer.
   * **Disadvantages**: May under-utilize deep layers’ capacity or over-parameterize early layers.

2. **Progressive (Increasing) Filters**

   * Start with few filters in early layers and **double** (or otherwise increase) them in deeper layers (e.g., 32→64→128).
   * **Advantages**: Mirrors hierarchical feature complexity (simple edges early, complex textures later); balances capacity/depth.
   * **Disadvantages**: More design choices; if growth is too aggressive, may bloat parameter count.

---

#### **A General Approach: Progressive Filters with Dropout**

1. **Start small** (e.g., 32 filters in the first conv block).
2. **Double** filters every block: 32 → 64 → 128 → …
3. **Use 3×3 kernels** throughout (standard best practice).
4. **Apply Dropout** (e.g., 0.25) after each pooling layer.
5. **Control depth** by how many times you double until reaching a target (e.g., 128–256 filters).

---

# Coding Example:

### 1. Fixed Filters Architecture with Tunable Head

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_fixed(
    num_blocks=3,
    num_filters=32,
    kernel_size=(3,3),
    pool_size=(2,2),
    activation='relu',
    conv_dropout=0.25,
    head_layers=1,
    head_units=128,
    head_dropout=0.5,
    optimizer='adam',
    learning_rate=0.001,
    input_shape=(64,64,3),
    num_classes=10
):
    model = Sequential()
    
    # Convolution + Pooling blocks
    for i in range(num_blocks):
        if i == 0:
            model.add(Conv2D(num_filters, kernel_size, activation=activation, padding='same',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, kernel_size, activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(conv_dropout))
    
    # Classification head (tunable number of Dense layers)
    model.add(Flatten())
    for _ in range(head_layers):
        model.add(Dense(head_units, activation=activation))
        model.add(Dropout(head_dropout))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage: 3 conv blocks, 32 filters, and a 2-layer head with 256 units each
model_fixed = build_cnn_fixed(num_blocks=3, num_filters=32, head_layers=2, head_units=256)
model_fixed.summary()
```

---

### 2. Progressive Filters Architecture (with fixed-head example)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_progressive(
    initial_filters=32,
    num_blocks=4,
    kernel_size=(3,3),
    pool_size=(2,2),
    activation='relu',
    conv_dropout=0.25,
    head_layers=1,
    head_units=256,
    head_dropout=0.5,
    optimizer='adam',
    learning_rate=0.001,
    input_shape=(64,64,3),
    num_classes=10
):
    model = Sequential()
    filters = initial_filters

    # Progressive Conv blocks
    for i in range(num_blocks):
        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation=activation, padding='same',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(filters, kernel_size, activation=activation, padding='same'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(conv_dropout))
        filters *= 2
    
    # Classification head
    model.add(Flatten())
    for _ in range(head_layers):
        model.add(Dense(head_units, activation=activation))
        model.add(Dropout(head_dropout))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage: start at 32 filters, 4 blocks, and a single-layer head of 128 units
model_prog = build_cnn_progressive(initial_filters=32, num_blocks=4, head_layers=1, head_units=128)
model_prog.summary()
```

---

# Training Process Hyperparameters

These control how the model learns from data.

* **Optimizer**: Algorithm for weight updates (e.g., **Adam**, **SGD**, **RMSprop**).
* **Learning Rate**: Step size for optimizer updates.
* **Batch Size**: Number of samples per gradient update.
* **Number of Epochs**: How many passes over the full training set.

---

# Hyperparameter Tuning

## 2.1. GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_cnn(
    num_blocks=3,
    initial_filters=32,
    kernel_size=3,
    activation='relu',
    conv_dropout=0.25,
    head_layers=1,
    head_units=128,
    head_dropout=0.5,
    optimizer='adam',
    learning_rate=0.001,
    batch_size=32,
    epochs=10
):
    return build_cnn_fixed(
        num_blocks=num_blocks,
        num_filters=initial_filters,
        kernel_size=(kernel_size,kernel_size),
        conv_dropout=conv_dropout,
        head_layers=head_layers,
        head_units=head_units,
        head_dropout=head_dropout,
        optimizer=optimizer,
        learning_rate=learning_rate
    )

cnn = KerasClassifier(build_fn=create_cnn, verbose=0)

param_grid = {
    'num_blocks': [2,3,4],
    'initial_filters': [16,32,64],
    'kernel_size': [3,5],
    'conv_dropout': [0.2,0.3],
    'head_layers': [1,2,3],
    'head_units': [64,128,256],
    'head_dropout': [0.3,0.5],
    'optimizer': ['adam','sgd'],
    'learning_rate': [1e-3,1e-4],
    'batch_size': [32,64],
    'epochs': [10,20]
}

grid = GridSearchCV(estimator=cnn, param_grid=param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
```

---

## 2.2. RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'num_blocks': randint(2,5),
    'initial_filters': randint(16,128),
    'kernel_size': [3,5],
    'conv_dropout': uniform(0.1,0.4),
    'head_layers': randint(1,4),
    'head_units': [64,128,256,512],
    'head_dropout': uniform(0.2,0.5),
    'optimizer': ['adam','sgd','rmsprop'],
    'learning_rate': uniform(1e-5,1e-2),
    'batch_size': [32,64,128],
    'epochs': randint(10,50)
}

rand_search = RandomizedSearchCV(
    estimator=cnn,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    n_jobs=-1
)
rand_search.fit(X_train, y_train)

print("Best Params:", rand_search.best_params_)
print("Best Score:", rand_search.best_score_)
```

---

## 2.3. Optuna

```python
import optuna
from sklearn.metrics import accuracy_score

def objective(trial):
    num_blocks      = trial.suggest_int('num_blocks', 2, 5)
    initial_filters = trial.suggest_categorical('initial_filters', [16,32,64,128])
    kernel_size     = trial.suggest_categorical('kernel_size', [3,5])
    conv_dropout    = trial.suggest_float('conv_dropout', 0.1, 0.5)
    head_layers     = trial.suggest_int('head_layers', 1, 3)
    head_units      = trial.suggest_categorical('head_units', [64,128,256,512])
    head_dropout    = trial.suggest_float('head_dropout', 0.2, 0.6)
    optimizer       = trial.suggest_categorical('optimizer', ['adam','sgd','rmsprop'])
    learning_rate   = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size      = trial.suggest_categorical('batch_size', [32,64,128])
    epochs          = trial.suggest_int('epochs', 10, 50)
    
    model = build_cnn_progressive(
        initial_filters=initial_filters,
        num_blocks=num_blocks,
        kernel_size=(kernel_size,kernel_size),
        conv_dropout=conv_dropout,
        head_layers=head_layers,
        head_units=head_units,
        head_dropout=head_dropout,
        optimizer=optimizer,
        learning_rate=learning_rate
    )
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)

print("Best Params:", study.best_params)
print("Best Accuracy:", study.best_value)
```

---
