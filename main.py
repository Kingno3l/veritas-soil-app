from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import joblib

# Initialize app
app = FastAPI(title="Soil SOC Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Your React port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- LOAD CNN MODEL AND SCALER --------
model = tf.keras.models.load_model("models/cnn/soil_soc_cnn.keras")
scaler = joblib.load("models/scaler.pkl")

# -------- INPUT SCHEMA --------
class SoilInput(BaseModel):
    Bulk_Density: float
    alt_BD: float
    Water_Content: float
    pH: float
    EC: float
    Soil_Respiration: float
    Bglucosidase: float
    Bglucosaminidase: float
    Alkaline_Phosphatase: float
    Acid_Phosphatase: float
    POX_C: float
    ACE: float

# -------- ROOT ENDPOINT --------
@app.get("/")
def root():
    return {"status": "Soil SOC API is running"}

# -------- PREDICTION ENDPOINT --------
@app.post("/predict")
def predict_soc(data: SoilInput):
    # Convert input to numpy array
    input_list = [
        data.Bulk_Density,
        data.alt_BD,
        data.Water_Content,
        data.pH,
        data.EC,
        data.Soil_Respiration,
        data.Bglucosidase,
        data.Bglucosaminidase,
        data.Alkaline_Phosphatase,
        data.Acid_Phosphatase,
        data.POX_C,
        data.ACE
    ]

    # Scale the features using training scaler
    input_scaled = scaler.transform([input_list])

    # Reshape for CNN: (1, n_features, 1)
    input_array = input_scaled.reshape((1, input_scaled.shape[1], 1))

    # Predict SOC
    prediction = model.predict(input_array)

    return {"predicted_SOC": float(prediction[0][0])}
