import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import numpy as np
import os

# Load prepared data
X_train = pd.read_csv("data/X_train.csv").values
X_test = pd.read_csv("data/X_test.csv").values
y_train = pd.read_csv("data/y_train.csv").values
y_test = pd.read_csv("data/y_test.csv").values

# Reshape data for CNN: (samples, features, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("CNN input shape:", X_train.shape)

# Build CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu",
           input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=64, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Train CNN
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate CNN
loss, mae = model.evaluate(X_test, y_test)
print(f"CNN Test MAE: {mae:.2f}")

# Save CNN model
os.makedirs("models/cnn", exist_ok=True)
# model.save("models/cnn/soil_soc_cnn")
model.save("models/cnn/soil_soc_cnn.keras")
print("CNN model saved successfully.")
