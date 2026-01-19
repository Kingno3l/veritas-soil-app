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
    Water_Content: float
    pH: float
    POX_C: float
    ACE: float
    Bglucosidase: float
    Bglucosaminidase: float
    Alkaline_Phosphatase: float
    Acid_Phosphatase: float
    Phosphodiesterase: float
    Arylsulfatase: float


# -------- ROOT ENDPOINT --------
@app.get("/")
def root():
    return {"status": "Soil SOC API is running"}

# -------- PREDICTION ENDPOINT --------
@app.post("/predict")
def predict_soc(data: SoilInput):
    input_list = [
        data.Bulk_Density,
        data.Water_Content,
        data.pH,
        data.POX_C,
        data.ACE,
        data.Bglucosidase,
        data.Bglucosaminidase,
        data.Alkaline_Phosphatase,
        data.Acid_Phosphatase,
        data.Phosphodiesterase,
        data.Arylsulfatase
    ]

    # Convert to numpy array for CNN
    input_array = np.array([input_list]).reshape(1, len(input_list), 1)

    prediction = model.predict(input_array)
    return {"predicted_SOC": float(prediction[0][0])}
