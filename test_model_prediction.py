import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/cnn/soil_soc_cnn.keras")

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

# Reshape X_test for CNN: (samples, features, 1)
X_test_array = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict
y_pred = model.predict(X_test_array)

# Convert predictions to 1D array
y_pred = y_pred.flatten()

# Calculate MAE
mae = np.mean(np.abs(y_test.values.flatten() - y_pred))
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# Optional: compare first 10 predictions vs actual
for i in range(10):
    print(f"Predicted: {y_pred[i]:.3f}, Actual: {y_test.values.flatten()[i]}")
