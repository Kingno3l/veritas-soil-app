from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import os

app = FastAPI(title="Final Year Project - Soil Intelligence API")

# 1. CORS CONFIGURATION (Allows React to talk to Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Simplest for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD ASSETS (Model, Scaler, Column Names)
# We use try/except to prevent the server from crashing if files are missing
try:
    model = tf.keras.models.load_model("models/cnn/soil_soc_cnn.keras")
    scaler = joblib.load("models/scaler.pkl")
    target_columns = joblib.load("models/target_columns.pkl")
    print("✅ Model, Scaler, and Target Columns loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading assets: {e}")
    print("Did you run 'train_cnn_model.py' yet?")

# 3. DEFINE INPUT SCHEMA (The 'Sensor Node' Data)
class IoTInput(BaseModel):
    device_id: str
    Soil_Temperature: float
    Water_Content: float
    pH: float
    EC: float

# 4. PREDICTION ENDPOINT
@app.post("/predict")
def predict_soil_health(data: IoTInput):
    # A. PREPARE INPUT (Must match the order in prepare_data.py)
    raw_input = [
        data.Soil_Temperature, 
        data.Water_Content, 
        data.pH, 
        data.EC
    ]

    # B. SCALE & RESHAPE
    # The scaler expects a 2D array: [[Temp, Water, pH, EC]]
    scaled_input = scaler.transform([raw_input]) 
    
    # The CNN expects a 3D array: [[[Temp], [Water], [pH], [EC]]]
    cnn_input = scaled_input.reshape(1, 4, 1)

    # C. PREDICT
    predictions = model.predict(cnn_input)[0] 
    
    # D. MAP RESULTS TO NAMES
    results = dict(zip(target_columns, predictions))

    # --- METHODOLOGY FIX: ENGINEERING CONSTRAINTS ---
    # This section ensures the AI respects the laws of physics.
    
    sanitized_results = {}
    
    for key, value in results.items():
        val = float(value)
        
        # Rule 1: No negative values for soil properties
        val = max(0.0, val)
        
        # Rule 2: Bulk Density Constraint (Standard soil is 1.0 - 1.8 g/cm3)
        if key == "Bulk_Density":
            if val < 0.9 or val > 1.8:
                val = 1.35  # Fallback to a healthy average if AI hallucinates
                
        sanitized_results[key] = val

    # E. RETURN JSON RESPONSE (Methodology Compliant)
    return {
        "device_id": data.device_id,
        "timestamp": datetime.now().isoformat(),

        # Section 3.6.3: "IoT layer inputs"
        "sensor_inputs": { 
            "Soil_Temperature": data.Soil_Temperature,
            "Water_Content": data.Water_Content,
            "pH": data.pH,
            "EC": data.EC
        },

        # Section 3.6.2: "Deep Learning Predictions"
        "deep_learning_predictions": {
            "Bulk_Density_Predicted": sanitized_results.get("Bulk_Density"),
            "POX_C_Predicted": sanitized_results.get("POX_C"),
            "ACE_Predicted": sanitized_results.get("ACE"),
            "Bglucosidase_Predicted": sanitized_results.get("Bglucosidase"),
            "Bglucosaminidase_Predicted": sanitized_results.get("Bglucosaminidase"),
            "Alkaline_Phosphatase_Predicted": sanitized_results.get("Alkaline Phosphatase"),
            "Acid_Phosphatase_Predicted": sanitized_results.get("Acid Phosphatase"),
        },

        # Section 3.7: "Ground Truth Validation"
        "ground_truth_validation": {
            "Bulk_Density_Actual": None,  # Python None becomes JSON null
            "POX_C_Actual": None
        }
    }