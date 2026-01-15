from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Initialize app
app = FastAPI(title="Soil SOC Prediction API")

# Load trained CNN model
# model = tf.keras.models.load_model("models/cnn/soil_soc_cnn.keras")

# Load the Simple Dense Model
model = tf.keras.models.load_model("models/soil_soc_model.keras")
# -------- INPUT SCHEMA --------
class SoilInput(BaseModel):
    Bulk_Density: float
    Water_Content: float
    SOC_pct: float
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
    # Convert input JSON to list in same order as training
    input_list = [
        data.Bulk_Density,
        data.Water_Content,
        data.SOC_pct,
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

    # Convert to 3D numpy array for CNN
    input_array = np.array([input_list]).reshape((1, len(input_list), 1))

    # Predict
    prediction = model.predict(input_array)

    return {"predicted_SOC": float(prediction[0][0])}
