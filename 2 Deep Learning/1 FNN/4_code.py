import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



# Load the Digits dataset
digits = load_digits()
X = digits.data  # Features (1,797 samples, 64 features each)
y = digits.target  # Labels (0-9 digits)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling to mean 0, std 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode the labels (since this is a classification task)
y_train_encoded = to_categorical(y_train, 10)  # Digits has 10 classes (0-9)
y_test_encoded = to_categorical(y_test, 10)



# FNN Model with Progressive Neurons
def build_fnn_progressive(initial_neurons=128, num_layers=3, activation='relu', dropout_rate=0.5, optimizer='adam', learning_rate=0.001):
    model = Sequential()

    # First layer with initial number of neurons
    model.add(Input(shape=(64,)))  # 64 features in the input
    model.add(Dense(initial_neurons, activation=activation))
    
    neurons = initial_neurons

    # Add progressively smaller hidden layers
    for _ in range(num_layers - 1):
        neurons = max(neurons // 2, 1)  # Halve the number of neurons (ensure it never goes below 1)
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))  # Add Dropout layer to prevent overfitting

    # Output layer (for classification task)
    model.add(Dense(10, activation='softmax'))  # 10 classes for Digits dataset

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







# Objective function for Optuna optimization with K-Fold Cross-Validation
def objective(trial):
    # Hyperparameters to tune
    num_layers = trial.suggest_int('num_layers', 2, 4)  # Number of hidden layers
    initial_neurons = trial.suggest_int('initial_neurons', 32, 128)  # Initial number of neurons in the first layer
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])  # Activation functions
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)  # Dropout rate for regularization
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])  # Optimizer choice
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Learning rate for optimization
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Batch size for training
    epochs = trial.suggest_int('epochs', 10, 50)  # Number of epochs for training
    
    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    
    # K-Fold Cross-Validation Loop
    for train_index, val_index in kf.split(X_train_scaled):
        # Split the data into training and validation sets for this fold
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train_encoded[train_index], y_train_encoded[val_index]
        
        # Build and train the model
        model = build_fnn_progressive(
            initial_neurons=initial_neurons, 
            num_layers=num_layers, 
            activation=activation, 
            dropout_rate=dropout_rate, 
            optimizer=optimizer, 
            learning_rate=learning_rate
        )
        
        model.fit(X_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, verbose=0)
        
        # Evaluate model performance on validation set
        y_pred = model.predict(X_val_fold)
        accuracy = accuracy_score(np.argmax(y_val_fold, axis=1), np.argmax(y_pred, axis=1))
        fold_accuracies.append(accuracy)
    
    # Return the average accuracy across all folds
    return np.mean(fold_accuracies)

# Create an Optuna study and optimize
study = optuna.create_study(direction='maximize')

# Run the optimization with cross-validation
study.optimize(objective, n_trials=20)

# Print the best hyperparameters and the corresponding accuracy

print("-----")
print("-----")
print("-----")
print("-----")
print("-----")


print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)






# Final model evaluation on the test set
# Manually extract the best model-building parameters
initial_neurons = study.best_params['initial_neurons']
num_layers = study.best_params['num_layers']
activation = study.best_params['activation']
dropout_rate = study.best_params['dropout_rate']
optimizer = study.best_params['optimizer']
learning_rate = study.best_params['learning_rate']

# Manually extract the best training parameters
batch_size = study.best_params['batch_size']
epochs = study.best_params['epochs']

# Build the final model using the best model-building parameters
final_model = build_fnn_progressive(
    initial_neurons=initial_neurons,
    num_layers=num_layers,
    activation=activation,
    dropout_rate=dropout_rate,
    optimizer=optimizer,
    learning_rate=learning_rate
)

# Train the final model using the best batch_size and epochs from Optuna
final_model.fit(X_train_scaled, y_train_encoded, 
                batch_size=batch_size, 
                epochs=epochs, 
                verbose=1)      # verbose=1: console e 1 ta progress bar dekhabe

# Evaluate test accuracy
test_loss, test_accuracy = final_model.evaluate(X_test_scaled, y_test_encoded, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")
