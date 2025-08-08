
# 🧠 RNN Model Architecture Hyperparameters

These define the structure of the **Recurrent Neural Network (RNN)**.

-   **Number of Recurrent Layers**: How many stacked RNN/GRU/LSTM layers.
    
-   **Hidden Size**: The number of units (neurons) in each RNN layer.
    
-   **Recurrent Cell Type**: The type of RNN cell used (`RNN`, `GRU`, `LSTM`).
    
-   **Bidirectionality**: Whether the model should read the sequence in both directions.
    
-   **Dropout**: Used between RNN layers or after RNN output to prevent overfitting.
    

----------

## 🔁 Hidden Size: Fixed vs Progressive

### 1. **Fixed Hidden Size**

-   Each RNN layer uses the **same number of hidden units**.
    
-   Good for general tasks with large datasets.
    
-   Risk of **overfitting** if layers are too wide and the dataset is small.
    

### 2. **Progressive Reduction**

-   The first RNN layer uses the largest number of hidden units.
    
-   Each subsequent layer uses **fewer units** (e.g., halved).
    
-   Helps compress temporal features and **regularize** the model.
    
-   Useful for **smaller datasets** or shallow sequential patterns.
    

----------

## ✅ General-Purpose Strategy (Recommended):

Use **progressive reduction with dropout**:

-   Layer 1: 128 units
    
-   Layer 2: 64 units
    
-   Layer 3: 32 units
    
-   Apply **Dropout** after each layer
    
-   Choose **LSTM** or **GRU** for complex time dependencies
    

----------

## 🧱 Coding Example: Progressive RNN Architecture

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dropout, Dense

# RNN model with progressive reduction in hidden size
def build_rnn_progressive(input_shape=(100, 10),  # e.g., 100 time steps, 10 features
                          initial_units=128,
                          num_layers=3,
                          cell_type='LSTM',
                          dropout_rate=0.3,
                          bidirectional=False,
                          output_units=1,
                          output_activation='sigmoid'):
    
    model = Sequential()
    RNNLayer = {'LSTM': LSTM, 'GRU': GRU, 'RNN': SimpleRNN}[cell_type]

    units = initial_units
    
    # First layer (with input shape)
    if bidirectional:
        model.add(tf.keras.layers.Bidirectional(RNNLayer(units, return_sequences=(num_layers > 1)), input_shape=input_shape))
    else:
        model.add(RNNLayer(units, return_sequences=(num_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    # Additional layers
    for i in range(1, num_layers):
        units = max(units // 2, 1)
        return_seq = i < num_layers - 1
        if bidirectional:
            model.add(tf.keras.layers.Bidirectional(RNNLayer(units, return_sequences=return_seq)))
        else:
            model.add(RNNLayer(units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(output_units, activation=output_activation))
    return model

```

----------

# ⚙️ Training Process Hyperparameters

These control **how the model learns** from the sequence data:

-   **Optimizer**: Method used to update the model weights.  
    _Best for RNNs:_ `Adam`, `RMSprop`
    
-   **Learning Rate**: Controls the step size for each weight update.
    
-   **Batch Size**: Number of sequence samples per training update.
    
-   **Number of Epochs**: Total passes through the training dataset.
    
-   **Loss Function**: Depends on task (`categorical_crossentropy` for classification, `MSE` for regression).
    

----------

## ✅ RNN Model Compilation

```python
model = build_rnn_progressive(input_shape=(100, 10), initial_units=128, num_layers=3, cell_type='LSTM')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

```

----------

# 🧪 Hyperparameter Tuning for RNNs

Use **GridSearchCV**, **RandomizedSearchCV**, or **Optuna** with `KerasClassifier`.

## ✅ 1. GridSearchCV

```python
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(cell_type='LSTM', initial_units=128, num_layers=2, dropout_rate=0.3, learning_rate=0.001):
    model = build_rnn_progressive(input_shape=(100, 10),
                                  cell_type=cell_type,
                                  initial_units=initial_units,
                                  num_layers=num_layers,
                                  dropout_rate=dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

param_grid = {
    'cell_type': ['LSTM', 'GRU'],
    'initial_units': [64, 128],
    'num_layers': [2, 3],
    'dropout_rate': [0.2, 0.5],
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

```

----------

## ✅ 2. RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'cell_type': ['LSTM', 'GRU'],
    'initial_units': randint(64, 256),
    'num_layers': randint(1, 4),
    'dropout_rate': uniform(0.2, 0.5),
    'learning_rate': uniform(1e-4, 1e-2),
    'batch_size': [16, 32, 64],
    'epochs': randint(10, 30)
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=30, cv=3, n_jobs=-1)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)

```

----------

## ✅ 3. Optuna for RNN

```python
import optuna
from sklearn.metrics import accuracy_score

def objective(trial):
    model = build_rnn_progressive(
        input_shape=(100, 10),
        cell_type=trial.suggest_categorical('cell_type', ['LSTM', 'GRU']),
        initial_units=trial.suggest_int('initial_units', 64, 256),
        num_layers=trial.suggest_int('num_layers', 1, 4),
        dropout_rate=trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    )

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train,
              batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
              epochs=trial.suggest_int('epochs', 10, 30),
              verbose=0)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred)
    
    return acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best Parameters:", study.best_params)
print("Best Accuracy:", study.best_value)

```

