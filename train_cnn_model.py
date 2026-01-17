import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

# STEP 1: Load prepared data (CSV)
X_train = pd.read_csv("data/X_train.csv").values
X_test = pd.read_csv("data/X_test.csv").values
y_train = pd.read_csv("data/y_train.csv").values
y_test = pd.read_csv("data/y_test.csv").values

# STEP 2: Reshape for CNN (samples, features, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("CNN input shape:", X_train.shape)

# STEP 3: Build CNN model
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

# STEP 4: Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# STEP 5: Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# STEP 6: Train
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# STEP 7: Evaluate
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test MAE: {mae:.2f}")

# STEP 8: Save model
os.makedirs("models/cnn", exist_ok=True)
model.save("models/cnn/soil_soc_cnn.keras")

print("CNN model trained and saved successfully.")
