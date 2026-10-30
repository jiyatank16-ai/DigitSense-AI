import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


# 1. Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize and Reshape to (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 3. Simple CNN Architecture
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# 4. Compile & Train
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Training simple CNN model...")
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

# 5. Save Model
model.save("digit_model.h5")
print("Model saved successfully as digit_model.h5")
