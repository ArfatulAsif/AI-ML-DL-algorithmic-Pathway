# --- 0. Import Libraries ---
```python
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
```

# --- 1. Dummy sequence classification dataset (Replace with real one) ---
```python
num_samples = 2000
time_steps = 10
num_features = 8
num_classes = 5

X = np.random.rand(num_samples, time_steps, num_features).astype(np.float32)
y = np.random.randint(0, num_classes, size=(num_samples,))
y_encoded = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
```

# --- 2. Build RNN model ---
```python
def build_rnn(
    rnn_type='LSTM',
    hidden_units=64,
    num_layers=1,
    bidirectional=False,
    dropout=0.3,
    input_shape=(10, 8),
    num_classes=5,
    optimizer='adam',
    learning_rate=0.001
):
    model = Sequential()

    RNNLayer = {'LSTM': LSTM, 'GRU': GRU, 'SimpleRNN': SimpleRNN}[rnn_type]

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        rnn_layer = RNNLayer(hidden_units, return_sequences=return_sequences)
        if bidirectional:
            rnn_layer = Bidirectional(rnn_layer)
        if i == 0:
            model.add(rnn_layer if not bidirectional else rnn_layer)
        else:
            model.add(rnn_layer)
        model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
```

# --- 3. Optuna objective function ---
```python
def objective(trial):
    # Model arch hyperparams
    rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU', 'SimpleRNN'])
    hidden_units = trial.suggest_int('hidden_units', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Training hyperparams
    
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 5, 15)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_rnn(
            rnn_type=rnn_type,
            hidden_units=hidden_units,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            input_shape=(time_steps, num_features),
            num_classes=num_classes,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
        preds = model.predict(X_val)
        acc = accuracy_score(np.argmax(y_val, axis=1), np.argmax(preds, axis=1))
        scores.append(acc)

    return np.mean(scores)
    
```
# --- 4. Run Optuna ---
```python
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best Hyperparameters:", study.best_params)
print("Best CV Accuracy:", study.best_value)
```

# --- 5. Final Model Evaluation ---
```python
final_model = build_rnn(
    rnn_type=study.best_params['rnn_type'],
    hidden_units=study.best_params['hidden_units'],
    num_layers=study.best_params['num_layers'],
    bidirectional=study.best_params['bidirectional'],
    dropout=study.best_params['dropout'],
    input_shape=(time_steps, num_features),
    num_classes=num_classes,
    optimizer=study.best_params['optimizer'],
    learning_rate=study.best_params['learning_rate']
)
final_model.fit(
    X_train, y_train,
    batch_size=study.best_params['batch_size'],
    epochs=study.best_params['epochs'],
    verbose=1
)
```

# Evaluate on test
```python
loss, acc = final_model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {acc:.4f}")
```
