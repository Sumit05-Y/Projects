import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Raw shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("dtype:", X_train.dtype)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("\nAfter reshape:")
print("X_train:", X_train.shape)

plt.imshow(X_train[0, :, :, 0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        units=128,
        activation="relu"
    ),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(
        units=10,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=128,
    verbose=1
)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

model.save("cnn_mnist_model.keras")

print("\nModel saved successfully!")
print("File Name: cnn_mnist_model.keras")

conv1_weights = model.layers[0].get_weights()[0]

print("Conv1 Weight Shape:", conv1_weights.shape)

fig, axes = plt.subplots(4, 8, figsize=(14, 8))

for i in range(32):
    ax = axes[i // 8, i % 8]
    filt = conv1_weights[:, :, 0, i]
    ax.imshow(filt, cmap="gray")
    ax.set_title(f"F{i}")
    ax.axis("off")

plt.suptitle("Learned Filters of First Convolution Layer")
plt.tight_layout()
plt.show()