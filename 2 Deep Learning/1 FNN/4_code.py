import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the MNIST image dataset
# We keep the original x_test to use for displaying the image later
(x_train, y_train), (x_test_orig, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess the image data
# Flatten the 28x28 images into a 1D vector of 784 pixels
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test_orig.reshape(10000, 784).astype("float32") / 255

# 3. Build the FNN model
model = keras.Sequential([
    # Input layer shape is 784 (the flattened image)
    layers.Input(shape=(784,)),
    # Hidden layer with 128 neurons
    layers.Dense(128, activation="relu"),

    # Output layer with 10 neurons (for digits 0-9)
    layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    # Use sparse_categorical_crossentropy for integer labels
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("--- Starting Model Training ---")
# 5. Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# 6. Evaluate the model on the entire test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy on {len(x_test)} images: {accuracy:.4f}")


# --- NEW SECTION: Extract, Display, and Predict a Single Image ---
print("\n--- Single Image Prediction ---")

# 7. Extract a single image to test
# Let's pick the first image from the original, non-flattened test set
image_to_test = x_test_orig[0]
label_to_test = y_test[0]

# 8. Display the image
plt.imshow(image_to_test, cmap='gray')
plt.title(f"Image to Classify (True Label: {label_to_test})")
plt.show()

# 9. Prepare the image for the model
# The model expects a batch of images, so we take our flattened test image
# and add a batch dimension. Shape changes from (784,) to (1, 784).
image_for_model = x_test[0]
image_for_model = np.expand_dims(image_for_model, axis=0)

# 10. Use the model to predict the class
prediction = model.predict(image_for_model)

# The prediction is an array of 10 probabilities.
# We find the index of the highest probability.
predicted_class = np.argmax(prediction)

print(f"\nModel's Prediction: {predicted_class}")
print(f"True Label: {label_to_test}")

if predicted_class == label_to_test:
    print("The model classified the image correctly!")
else:
    print("The model made a mistake.")

