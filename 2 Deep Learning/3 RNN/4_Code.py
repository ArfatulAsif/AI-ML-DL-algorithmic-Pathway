import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Load and preprocess IMDB dataset (reduced size for faster training) ---
print("Loading and preprocessing IMDB dataset...")
max_features = 5000  # Vocabulary size (top 5000 words)
maxlen = 200         # Max length of review sequences

(X_train_full, y_train_full), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Reduce dataset size
X_train_full = X_train_full[:3000]
y_train_full = y_train_full[:3000]
X_test = X_test[:1000]
y_test = y_test[:1000]

X_train = pad_sequences(X_train_full, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
y_train = np.array(y_train_full)
y_test = np.array(y_test)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# --- 2. Build RNN model ---
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
    print(f"\nBuilding {rnn_type} model: layers={num_layers}, units={hidden_units}, bidirectional={bidirectional}, embedding_dim={embedding_dim}")
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

# --- 3. Optuna objective function ---
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

    print("Trial params:", trial.params)

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

    mean_acc = np.mean(scores)
    print(f"Trial Mean Accuracy: {mean_acc:.4f}")
    return mean_acc

# --- 4. Run Optuna ---
print("\nRunning Optuna hyperparameter optimization...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("\nBest Hyperparameters:", study.best_params)
print("Best CV Accuracy:", study.best_value)

# --- 5. Final Model Evaluation ---
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
