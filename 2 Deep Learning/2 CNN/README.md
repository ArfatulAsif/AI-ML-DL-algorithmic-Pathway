# CNN Model implementation

---

# **1. Data Preprocessing & Augmentation**

We load **CIFAR-10** (subset of 2,000 samples), scale pixel values, one-hot encode labels, then set up **data augmentation** (rotation, shifts, flips) with tunable hyperparameters.

```python
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 and subset
(X_full, y_full), _ = cifar10.load_data()
X = X_full[:2000].astype('float32') / 255.0
y = y_full[:2000].flatten()

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encode labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded  = to_categorical(y_test, 10)

# --- Augmentation hyperparameters (to be tuned) ---
rotation_range      = 20      # degrees
width_shift_range   = 0.1     # fraction of width
height_shift_range  = 0.1     # fraction of height
horizontal_flip     = True    # boolean

train_datagen = ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    horizontal_flip=horizontal_flip
)
# Note: No augmentation on test set
```

**Explanation:**

* **Scaling**: Normalize pixel values to \[0,1].
* **One-hot Encoding**: Convert labels into binary class vectors.
* **Augmentation**: Random rotation, shifting, and flipping to improve robustness.

---

# **2. Build the CNN Model (Progressive Filters & Tunable Head)**

We define a **progressive-filters** CNN where **all** architecture hyperparameters are passed in: convolution blocks, filter sizes, strides, padding, pooling, dropout, and a tunable dense head.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_cnn_progressive(
    # Conv-block hyperparameters
    initial_filters=32,
    num_blocks=3,
    kernel_size=(3,3),
    strides=(1,1),
    padding='same',                # 'same' or 'valid'
    pooling_type='max',            # 'max' or 'avg'
    pool_size=(2,2),
    activation='relu',
    conv_dropout=0.25,
    # Head hyperparameters
    head_layers=1,
    head_units=128,
    head_dropout=0.5,
    # Optimization hyperparameters
    optimizer='adam',
    learning_rate=0.001,
    input_shape=(32,32,3),
    num_classes=10
):
    model = Sequential()
    model.add(Input(shape=input_shape))
    filters = initial_filters

    # Progressive conv → pool → dropout blocks
    for _ in range(num_blocks):
        model.add(Conv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            activation=activation
        ))
        if pooling_type == 'max':
            model.add(MaxPooling2D(pool_size=pool_size))
        else:
            model.add(AveragePooling2D(pool_size=pool_size))
        model.add(Dropout(conv_dropout))
        filters *= 2

    # Flatten → tunable head
    model.add(Flatten())
    for _ in range(head_layers):
        model.add(Dense(head_units, activation=activation))
        model.add(Dropout(head_dropout))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

**Explanation:**

* **Strides & Padding**: Control feature map resolution.
* **Pooling Type & Size**: Choose Max or Average pooling.
* **Progressive Filters**: Doubles filters each block.
* **Conv & Head Dropout**: Regularize conv blocks and dense layers.
* **Tunable Head**: `head_layers`, `head_units`.
* **Optimizer & LR**: Flexibly chosen per hyperparameter.

---

# **3. Hyperparameter Tuning with Optuna (3-Fold CV & Augmentation)**

We tune **every** hyperparameter—from conv blocks, pooling, strides, head depth, to augmentation settings—using Optuna with **3-fold cross-validation**.

```python
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def objective(trial):
    # Architecture
    num_blocks      = trial.suggest_int('num_blocks', 2, 4)
    initial_filters = trial.suggest_int('initial_filters', 16, 64, step=16)
    kernel          = trial.suggest_categorical('kernel_size', [3,5])
    strides         = trial.suggest_categorical('strides', [1,2])
    padding         = trial.suggest_categorical('padding', ['same','valid'])
    pooling_type    = trial.suggest_categorical('pooling_type', ['max','avg'])
    pool            = trial.suggest_categorical('pool_size', [2,3])
    conv_dropout    = trial.suggest_float('conv_dropout', 0.1, 0.5)

    # Head
    head_layers     = trial.suggest_int('head_layers', 1, 3)
    head_units      = trial.suggest_categorical('head_units', [64,128,256,512])
    head_dropout    = trial.suggest_float('head_dropout', 0.2, 0.6)

    # Training
    optimizer       = trial.suggest_categorical('optimizer', ['adam','sgd','rmsprop'])
    learning_rate   = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size      = trial.suggest_categorical('batch_size', [16,32,64])
    epochs          = trial.suggest_int('epochs', 5, 20)

    # Augmentation
    rotation_range     = trial.suggest_int('rotation_range', 0, 40)
    width_shift_range  = trial.suggest_float('width_shift_range', 0.0, 0.2)
    height_shift_range = trial.suggest_float('height_shift_range', 0.0, 0.2)
    horizontal_flip    = trial.suggest_categorical('horizontal_flip', [True, False])

    # 3-Fold CV
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]

        # Datagen per trial
        datagen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip
        )
        datagen.fit(X_tr)

        # Build & train
        model = build_cnn_progressive(
            initial_filters=initial_filters,
            num_blocks=num_blocks,
            kernel_size=(kernel,kernel),
            strides=(strides,strides),
            padding=padding,
            pooling_type=pooling_type,
            pool_size=(pool,pool),
            conv_dropout=conv_dropout,
            head_layers=head_layers,
            head_units=head_units,
            head_dropout=head_dropout,
            optimizer=optimizer,
            learning_rate=learning_rate
        )
        model.fit(
            datagen.flow(X_tr, y_tr, batch_size=batch_size),
            epochs=epochs,
            verbose=0
        )

        # Evaluate
        preds = model.predict(X_val)
        acc = accuracy_score(
            np.argmax(y_val,axis=1),
            np.argmax(preds,axis=1)
        )
        scores.append(acc)

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
print("Best Hyperparameters:", study.best_params)
print("Best CV Accuracy:   ", study.best_value)
```

**Explanation:**

* **All architecture hyperparameters** now tunable: blocks, filters, kernel, strides, padding, pooling.
* **Head depth/width** and **dropout** are optimized.
* **Augmentation parameters** included.
* **3-fold CV** gives robust estimates.

---

# **4. Final Model Evaluation**

Rebuild with the best settings, train on the full 1,600-sample training set (with augmentation), and evaluate on the 400-sample test set.

```python
# Extract best params
best = study.best_params

# Final datagen
final_datagen = ImageDataGenerator(
    rotation_range=best['rotation_range'],
    width_shift_range=best['width_shift_range'],
    height_shift_range=best['height_shift_range'],
    horizontal_flip=best['horizontal_flip']
)
final_datagen.fit(X_train)

# Build final model
final_model = build_cnn_progressive(
    initial_filters=best['initial_filters'],
    num_blocks=best['num_blocks'],
    kernel_size=(best['kernel_size'], best['kernel_size']),
    strides=(best['strides'], best['strides']),
    padding=best['padding'],
    pooling_type=best['pooling_type'],
    pool_size=(best['pool_size'], best['pool_size']),
    conv_dropout=best['conv_dropout'],
    head_layers=best['head_layers'],
    head_units=best['head_units'],
    head_dropout=best['head_dropout'],
    optimizer=best['optimizer'],
    learning_rate=best['learning_rate']
)

# Train on full training set
final_model.fit(
    final_datagen.flow(X_train, y_train_encoded, batch_size=best['batch_size']),
    epochs=best['epochs'],
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = final_model.evaluate(X_test, y_test_encoded, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")
```

**Explanation:**

* Uses **full training set** with optimized augmentation and hyperparameters.
* Reports final **test accuracy** on the hold-out set.

---



**Caution about errors:** 
**You might get a negative dimension error.**
```
This might happen if too many pooling layers drastically reduce the input resolution. With an input of 32x32 and 4 blocks, pooling reduces the dimensions progressively (16, 8, 4, 2). After 4 blocks, you end up with 2x2, followed by a Conv2D layer. If padding is “same”, it keeps dimensions, but if the kernel size is too large, it could cause issues. It seems like using "valid" padding for the convolution could help prevent this.
```

**To fix this issue, adjust the `pooling` or related hyperparameters**. 

At `4_code.py` the correct code for running is given. 


