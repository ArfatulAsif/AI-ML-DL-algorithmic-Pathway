Of course. Here are the common hyperparameters for a Feedforward Neural Network (FNN).

---
## Model Architecture Hyperparameters
These define the structure of the network itself.

-   **Number of Hidden Layers**: The **depth** of the network (how many layers are stacked between the input and output).
-   **Number of Neurons per Layer**: The **width** of each hidden layer.
-   **Activation Functions**: The non-linear function used in the hidden layers (e.g., **ReLU**, **Tanh**, **Leaky ReLU**).
-   **Dropout Rate (dropout_rate):**  `Regularization` Used to prevent overfitting by randomly setting some neuron outputs to zero during training.

---
## Training Process Hyperparameters
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
