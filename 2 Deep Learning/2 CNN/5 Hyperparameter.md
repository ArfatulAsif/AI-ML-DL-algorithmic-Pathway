# Model Architecture Hyperparameters

These define the structure of the CNN itself — how many layers it has, what kind of operations are used, and how wide or deep it goes.

* **Number of Convolutional Layers**: The number of stacked convolution blocks used to extract features from the input.
* **Number of Filters per Layer**: Determines how many patterns the network learns at each level. More filters = more feature maps = more expressive power.
* **Kernel Size (kernel\_size)**: The spatial size of each filter (e.g., 3×3, 5×5). Smaller kernels learn local features, larger ones see more context.
* **Stride**: How much the kernel slides when scanning the image. A larger stride leads to smaller output and faster computation.
* **Padding**: Whether to add zero-padding around the image to preserve spatial size. `"valid"` means no padding; `"same"` keeps dimensions constant.
* **Pooling Type and Size**: Typically max-pooling (e.g., 2×2) to reduce dimensionality while preserving dominant features.
* **Activation Functions**: Usually **ReLU**, but also **Leaky ReLU**, **ELU**, or **Swish** in advanced models.
* **Dropout Rate (dropout\_rate)**: `Regularization` technique to prevent overfitting by randomly disabling some neurons during training.
* **Batch Normalization**: Used after Conv layers to stabilize training and improve generalization.

---

### **Fixed Filters vs. Progressive Filters: A Brief Note**

#### **1. Fixed Filter Size and Count**:

* In this approach, each convolutional layer has the **same number of filters** and same kernel size (e.g., all 64 filters, 3×3 kernels).
* This creates a consistent feature extraction pipeline.
* **Advantages**: Easier to design, good for general-purpose models.
* **Disadvantages**: Can be inefficient or too simple for deeper networks.

#### **2. Progressive Design (Increasing Filters)**:

* As we go deeper, we **increase the number of filters** (e.g., 32 → 64 → 128) to allow learning of more abstract features.
* Often used in real CNN architectures like VGG, ResNet, etc.
* **Advantages**: Efficient use of resources. Shallow layers detect edges; deeper layers learn shapes and objects.
* **Disadvantages**: Slightly more complex design, needs tuning for best performance.

---

### **A General Approach: Progressive Filters + Dropout + BatchNorm**

A commonly used structure follows this pattern:

* **Start with fewer filters** (e.g., 32) in the first layer to detect low-level features.
* Increase filters progressively:

  * First Conv block: 32 filters
  * Second: 64 filters
  * Third: 128 filters, and so on
* Use **3×3 kernels**, **stride=1**, and **same padding**.
* Apply **MaxPooling** (2×2) after each block to reduce size.
* Add **Dropout** and **BatchNorm** to improve regularization.
* Flatten and pass to dense layers (typically 1–2) before the output.

---

### **Coding Example: Progressive CNN Architecture**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# CNN with Progressive Filters
def build_cnn(progressive_filters=[32, 64, 128], kernel_size=(3, 3), dropout_rate=0.5, activation='relu', input_shape=(28, 28, 1), output_classes=10):
    model = Sequential()

    for i, filters in enumerate(progressive_filters):
        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation=activation, padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(filters, kernel_size, activation=activation, padding='same'))
        
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_classes, activation='softmax'))  # e.g., 10 classes for MNIST

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build example CNN
model = build_cnn()
model.summary()
```

In this model:

* Filters grow progressively from 32 → 64 → 128.
* Dropout and BatchNorm are used for regularization.
* MaxPooling shrinks the image size after each Conv block.
* A fully connected layer is used before the final output layer.

---

### **Summary**

* **Fixed Filters**: Easy to design, consistent structure. Good for simpler problems.
* **Progressive Filters**: More powerful, commonly used in real-world CNNs. Helps model deeper patterns layer by layer.
* **Use BatchNorm and Dropout** for better generalization.
* Start small (e.g., 2–3 blocks) and grow based on model performance.

---

# Training Process Hyperparameters

These control **how the CNN learns** from the data during training.

* **Optimizer**: The algorithm that adjusts weights based on gradients.
  *Examples:*

  * **Adam** (default and adaptive)
  * **SGD** (classic, with optional momentum)
  * **RMSprop** (great for sequence data and noisy gradients)

* **Learning Rate**: The step size for weight updates.
  Too high = unstable. Too low = slow convergence.

* **Batch Size**: How many images are used in one training step.
  Larger batch → faster but more memory.
  Smaller batch → noisier updates but better generalization.

* **Number of Epochs**: How many full passes through the entire training set.
  Use early stopping if validation performance worsens.

* **Loss Function**:

  * **categorical\_crossentropy** (multi-class classification)
  * **binary\_crossentropy** (binary classification)
  * **mse / mae** (for regression tasks)

---

### **1. Building a Progressive CNN with Training Params**

```python
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_trainable_cnn(progressive_filters=[32, 64], dropout_rate=0.5, activation='relu', optimizer='adam', learning_rate=0.001, input_shape=(28,28,1), output_classes=10):
    model = Sequential()
    
    for i, filters in enumerate(progressive_filters):
        if i == 0:
            model.add(Conv2D(filters, (3,3), activation=activation, padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(filters, (3,3), activation=activation, padding='same'))
        
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_classes, activation='softmax'))

    # Choose optimizer
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

---

### **When to Use What**

* **Adam**: Best general-purpose optimizer. Default choice.
* **SGD**: Better when paired with momentum and scheduled learning rate.
* **RMSprop**: Best for non-stationary problems (e.g., RNNs, time-varying inputs).
* **Small Learning Rate + Many Epochs**: Needed for deep CNNs.
* **Dropout**: Always recommended after dense layers and sometimes after Conv blocks.

