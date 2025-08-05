
# 🧠 CNN Model Implementation

# **1. Data Preprocessing:**

We’ll reshape the flat Digits dataset into 2D image format suitable for CNNs (8×8 images), normalize pixel values, and **one-hot encode** the labels for classification.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the Digits dataset
digits = load_digits()
X = digits.images  # 8x8 grayscale images (1,797 samples)
y = digits.target  # Labels (0-9)

# Normalize pixel values (0 to 1)
X = X / 16.0  # Pixel range is originally 0–16

# Reshape to (n_samples, height, width, channels)
X = X.reshape(-1, 8, 8, 1)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)
```

**Explanation:**

* **Reshape for CNN**: We convert the 64-feature vector into an 8×8×1 image (1 = grayscale channel).
* **Normalization**: Dividing by 16 scales pixel values to 0–1.
* **One-hot encoding**: Prepares the target labels for multi-class classification.

---

# **2. Build the CNN Model (Progressive Filters):**

Here we define a simple CNN using the **progressive filter strategy** (e.g., 32 → 64 → 128 filters). We apply **Dropout** and **BatchNormalization** to help prevent overfitting and stabilize training.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input

# CNN Model with Progressive Filters
def build_cnn(progressive_filters=[32, 64], kernel_size=(3,3), dropout_rate=0.3, activation='relu', optimizer='adam', learning_rate=0.001, input_shape=(8, 8, 1), num_classes=10):
    model = Sequential()
    
    model.add(Input(shape=input_shape))
    
    for i, filters in enumerate(progressive_filters):
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(64, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer selection
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

**Explanation:**

* **Progressive Filters**: More filters in deeper layers help extract more complex features.
* **Kernel Size**: 3×3 is the most common and effective choice for small images.
* **Dropout & BatchNorm**: Combat overfitting and stabilize learning.
* **Dense Layers**: One dense layer before the final softmax to combine extracted features.

---

# **3. Hyperparameter Tuning with Optuna (Using K-Fold Cross-Validation):**

We tune model architecture and training hyperparameters using Optuna, including: number of filters, activation, dropout, optimizer, learning rate, batch size, and epochs — using **5-fold cross-validation** for robust evaluation.

```python
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Optuna objective with CNN + K-Fold CV
def objective(trial):
    # Hyperparameters
    num_filters_1 = trial.suggest_categorical('num_filters_1', [16, 32, 64])
    num_filters_2 = trial.suggest_categorical('num_filters_2', [32, 64, 128])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 40)

    # Use 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train_encoded[train_index], y_train_encoded[val_index]

        model = build_cnn(
            progressive_filters=[num_filters_1, num_filters_2],
            activation=activation,
            dropout_rate=dropout_rate,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

        model.fit(X_fold_train, y_fold_train, batch_size=batch_size, epochs=epochs, verbose=0)
        preds = model.predict(X_fold_val)
        acc = accuracy_score(np.argmax(y_fold_val, axis=1), np.argmax(preds, axis=1))
        scores.append(acc)

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Best parameters and performance
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)
```

**Explanation:**

* **Objective Function**: Builds and trains a CNN with the trial’s hyperparameters.
* **Cross-validation**: 5-fold CV prevents overfitting on one split and gives a more reliable accuracy estimate.
* **Optuna Search Space**: Flexible search over filters, activations, optimizers, and training params.

---

# **4. Model Evaluation:**

After finding the best hyperparameters, we retrain the model on the **entire training set** and evaluate on the **test set**.

```python
# Retrieve best params from Optuna study
best = study.best_params
final_model = build_cnn(
    progressive_filters=[best['num_filters_1'], best['num_filters_2']],
    activation=best['activation'],
    dropout_rate=best['dropout_rate'],
    optimizer=best['optimizer'],
    learning_rate=best['learning_rate']
)

# Train on full training set
final_model.fit(X_train, y_train_encoded, batch_size=best['batch_size'], epochs=best['epochs'], verbose=1)

# Evaluate on test set
test_loss, test_acc = final_model.evaluate(X_test, y_test_encoded, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")
```

**Explanation:**

* **Final Training**: Uses all training data and Optuna-optimized configuration.
* **Test Evaluation**: Evaluates how well the tuned model generalizes to unseen data.

