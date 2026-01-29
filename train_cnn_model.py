import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, Dropout
import os

# 1. Load Prepared Data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# 2. Reshape for CNN (Samples, Features, 1)
# CNNs need 3 dimensions: [Rows, 4 Columns, 1 Channel]
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

input_shape = (4, 1)  # 4 Sensors
output_units = y_train.shape[1] # 7 Predictions

# 3. Build Model (Matches Section 3.6.2)
model = Sequential([
    Input(shape=input_shape),
    
    # Feature Extraction
    Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
    Dropout(0.2),
    Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'),
    Flatten(),
    
    # Reasoning
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    
    # Output (Regression)
    Dense(output_units, activation='linear') 
])

# 4. Compile & Train
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training Neural Network...")
model.fit(
    X_train_reshaped, y_train,
    epochs=50, 
    batch_size=16,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)

# 5. Save Model
os.makedirs("models/cnn", exist_ok=True)
model.save("models/cnn/soil_soc_cnn.keras")
print("Model Trained & Saved.")