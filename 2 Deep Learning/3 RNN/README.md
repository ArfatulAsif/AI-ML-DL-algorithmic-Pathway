Perfect! Here's your complete **structured RNN implementation**—formatted exactly like your previous FNN structure—with detailed sections and beginner-friendly explanations, as a best teacher would present:

---

# 🧠 **RNN Model Implementation Using Optuna and IMDB Dataset**

---

## **1. Data Preprocessing:**

We use the **IMDB Movie Review Dataset**, containing **sentiment-labeled reviews** (positive or negative). The reviews are tokenized as sequences of word indices and padded to the same length for input consistency.

```python
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
max_features = 5000  # Only use top 5000 frequent words
maxlen = 200         # Maximum length of a review (truncate/pad)

# Load IMDB dataset
(X_train_full, y_train_full), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Reduce data size for faster training
X_train_full = X_train_full[:3000]
y_train_full = y_train_full[:3000]
X_test = X_test[:1000]
y_test = y_test[:1000]

# Pad sequences (make all reviews the same length)
X_train = pad_sequences(X_train_full, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
y_train = np.array(y_train_full)
y_test = np.array(y_test)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
```

### 🧾 Explanation:

* **max\_features**: Keeps only the top 5000 most frequent words in the vocabulary.
* **maxlen**: Ensures all input sequences are of equal length (200 tokens).
* **pad\_sequences**: Truncates longer reviews and pads shorter ones with zeros.
* **IMDB Dataset**: Contains binary sentiment labels (0 = negative, 1 = positive).

---

## **2. Build the RNN Model (Customizable Layers):**

Here, we build an RNN model using **LSTM**, **GRU**, or **SimpleRNN**, with support for **bidirectional layers**, **dropout**, and different optimizers.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_rnn(
    rnn_type='LSTM',
    hidden_units=64,
    num_layers=1,
    bidirectional=False,
    dropout=0.3,
    embedding_dim=128,
    input_length=200,
    optimizer='adam',
    learning_rate=0.001
):
    print(f"\nBuilding {rnn_type} model...")
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=input_length))

    RNNLayer = {'LSTM': LSTM, 'GRU': GRU, 'SimpleRNN': SimpleRNN}[rnn_type]

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        rnn_layer = RNNLayer(hidden_units, return_sequences=return_sequences)
        if bidirectional:
            rnn_layer = Bidirectional(rnn_layer)
        model.add(rnn_layer)
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
```

### 🧾 Explanation:

* **Embedding Layer**: Transforms word indices into dense vectors.
* **RNN Type**: Choose from `LSTM`, `GRU`, or `SimpleRNN`.
* **Bidirectional**: Helps the model understand context from both past and future.
* **Dropout**: Prevents overfitting.
* **Optimizer**: Choose between `Adam`, `SGD`, and `RMSprop`.

---

## **3. Hyperparameter Tuning with Optuna (K-Fold CV):**

We use **Optuna** to tune hyperparameters like layer type, dropout, learning rate, bidirectionality, etc., using **3-Fold Cross-Validation** for reliable evaluation.

```python
import optuna
from sklearn.model_selection import KFold

def objective(trial):
    print("\nStarting new Optuna trial...")
    rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU', 'SimpleRNN'])
    hidden_units = trial.suggest_int('hidden_units', 32, 64)
    num_layers = trial.suggest_int('num_layers', 1, 2)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128])
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    epochs = trial.suggest_int('epochs', 2, 5)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/3")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = build_rnn(
            rnn_type=rnn_type,
            hidden_units=hidden_units,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            embedding_dim=embedding_dim,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
        acc = model.evaluate(X_val, y_val, verbose=0)[1]
        print(f"    Fold Accuracy: {acc:.4f}")
        scores.append(acc)

    return np.mean(scores)
```

```python
print("\nRunning Optuna hyperparameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("\nBest Hyperparameters:", study.best_params)
print("Best CV Accuracy:", study.best_value)
```

### 🧾 Explanation:

* **Optuna**: Smart hyperparameter optimization library.
* **KFold**: Each trial is validated across 3 different data splits.
* **Trial Parameters**: Includes model type, number of layers, dropout, etc.
* **Accuracy**: We return the average accuracy over all folds.

---

## **4. Final Model Evaluation:**

We use the **best hyperparameters** from Optuna to train on the full training set and evaluate on the test set.

```python
print("\nTraining final model with best hyperparameters...")
final_model = build_rnn(
    rnn_type=study.best_params['rnn_type'],
    hidden_units=study.best_params['hidden_units'],
    num_layers=study.best_params['num_layers'],
    bidirectional=study.best_params['bidirectional'],
    dropout=study.best_params['dropout'],
    embedding_dim=study.best_params['embedding_dim'],
    optimizer=study.best_params['optimizer'],
    learning_rate=study.best_params['learning_rate']
)

final_model.fit(
    X_train, y_train,
    batch_size=study.best_params['batch_size'],
    epochs=study.best_params['epochs'],
    verbose=1
)

loss, acc = final_model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {acc:.4f}")
```

### 🧾 Explanation:

* **Best Model**: Trained using full training data and Optuna-optimized settings.
* **Test Accuracy**: Measures final performance on unseen test reviews.

