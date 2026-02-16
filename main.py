from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime
import os

# Initialize the App
app = FastAPI(title="Final Year Project - Soil Intelligence API")

# 1. CORS CONFIGURATION
# Allows your React Frontend (Vercel/Localhost) to talk to this Python Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Simplest for development/demo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD ASSETS (Model, Scaler, Column Names)
# We use try/except to prevent the server from crashing if files are missing
try:
    # Load the trained CNN model
    model = tf.keras.models.load_model("models/cnn/soil_soc_cnn.keras")
    # Load the scaler used during training
    scaler = joblib.load("models/scaler.pkl")
    # Load the list of target names (POX-C, Bulk Density, etc.)
    target_columns = joblib.load("models/target_columns.pkl")
    print("✅ Model, Scaler, and Target Columns loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading assets: {e}")
    print("Did you run 'train_cnn_model.py' yet?")

# 3. DEFINE INPUT SCHEMA (The 'Sensor Node' Data)
# This includes both the Physics variables (for AI) and NPK (for Rule-Engine)
class SoilInput(BaseModel):
    device_id: str
    Soil_Temperature: float
    Water_Content: float
    pH: float
    EC: float
    # Hybrid Addition: NPK Inputs (Rule-Based)
    Nitrogen: float
    Phosphorus: float
    Potassium: float

# 4. PREDICTION ENDPOINT
@app.post("/predict")
async def predict_soil_health(data: SoilInput):
    try:
        # --- PATH A: DEEP LEARNING (Biological Health) ---
        
        # 1. Prepare Input for AI (Only use the 4 physics variables)
        # Note: We purposely exclude NPK here because the CNN wasn't trained on them.
        raw_features = [
            data.Soil_Temperature, 
            data.Water_Content, 
            data.pH, 
            data.EC
        ]

        # 2. Scale the data (0 to 1 range) using the loaded scaler
        # The scaler expects a 2D array: [[Temp, Water, pH, EC]]
        scaled_features = scaler.transform([raw_features])
        
        # 3. Reshape for CNN (Samples, TimeSteps, Features) -> (1, 4, 1)
        # This matches the 'Conv1D' input shape expected by the model
        cnn_input = scaled_features.reshape(1, 4, 1)

        # 4. Predict
        predictions = model.predict(cnn_input)[0] 
        
        # 5. Map results to names and Apply Physics Constraints
        ai_results = {}
        target_names = target_columns if 'target_columns' in globals() else [
            "Bulk_Density", "POX_C", "Bglucosidase", "Soil_Respiration", 
            "ACE_Protein", "Alkaline_Phosphatase", "Acid_Phosphatase", "Bglucosaminidase"
        ]

        # Physics-Constraint Layer (Safety Logic)
        for i, name in enumerate(target_names):
            val = float(predictions[i])
            
            # Constraint 1: No negative values for biological properties
            val = max(0.0, val)
            
            # Constraint 2: Bulk Density Clamp (Standard soil is 0.9 - 1.8 g/cm3)
            if name == "Bulk_Density":
                if val < 0.9 or val > 1.8:
                    val = 1.35  # Reset to a safe average if AI hallucinates
            
            ai_results[name] = val

        # --- PATH B: RULE-BASED ENGINE (Fertility Status) ---
        
        fertility_status = "Balanced (Optimal)"
        fertility_reason = "Nutrient levels are within healthy ranges."
        
        # Simple Agronomic Thresholds
        if data.Nitrogen < 50 or data.Phosphorus < 20 or data.Potassium < 40:
            fertility_status = "Low Fertility (Deficient)"
            fertility_reason = "Detected deficiencies in primary macronutrients."
        elif data.Nitrogen > 200:
            fertility_status = "Excessive Nitrogen"
            fertility_reason = "Nitrogen levels are too high, risk of toxicity."

        # --- PATH C: OVERALL SYSTEM STATUS (The Green/Orange Card Logic) ---
        
        overall_health = "Good (Healthy)"
        
        # Logic: If Bulk Density is bad OR Fertility is low -> Warning
        bd_val = ai_results.get("Bulk_Density", 1.3)
        
        if bd_val > 1.6 or bd_val < 0.9: 
            overall_health = "Attention Needed (Compaction Risk)"
        elif fertility_status == "Low Fertility (Deficient)":
            overall_health = "Attention Needed (Nutrient Deficient)"

        # 5. RETURN FINAL JSON
        return {
            "device_id": data.device_id,
            "timestamp": datetime.now().isoformat(),
            
            # The Main Dashboard Status
            "status": overall_health,
            "fertility_analysis": fertility_status,
            "fertility_reason": fertility_reason,
            
            # The AI Predictions
            "predictions": ai_results,
            
            # Echo the inputs back (for validation)
            "inputs": {
                "Physics": {
                    "Temp": data.Soil_Temperature,
                    "Moisture": data.Water_Content,
                    "pH": data.pH,
                    "EC": data.EC
                },
                "Nutrients": {
                    "N": data.Nitrogen,
                    "P": data.Phosphorus,
                    "K": data.Potassium
                }
            }
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Prediction failed. Check server logs.", "details": str(e)}

# 5. ROOT ENDPOINT (Just to check if server is running)
@app.get("/")
def home():
    return {"message": "Soil Intelligence API is Online 🟢", "docs_url": "/docs"}