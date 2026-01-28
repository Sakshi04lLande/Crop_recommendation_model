from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# ================================
# Load Model & Encoder
# ================================
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

# ================================
# FastAPI App
# ================================
app = FastAPI(
    title="Crop Recommendation API",
    description="ML-based crop recommendation system",
    version="1.0"
)

# ================================
# Input Schema
# ================================
class CropInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# ================================
# Root Endpoint
# ================================
@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running 🌱"}

# ================================
# Single Crop Prediction
# ================================
@app.post("/predict")
def predict_crop(data: CropInput):

    input_df = pd.DataFrame([{
    "Nitrogen": data.Nitrogen,
    "Phosphorus": data.Phosphorus,
    "Potassium": data.Potassium,
    "temperature": data.temperature,
    "humidity": data.humidity,
    "ph": data.ph,
    "rainfall": data.rainfall
}])

    prediction = model.predict(input_df)
    crop = encoder.inverse_transform(prediction)[0]

    return {
        "recommended_crop": crop
    }

# ================================
# Top-3 Crop Recommendation
# ================================
@app.post("/recommend")
def recommend_top_3(data: CropInput):

    input_df = pd.DataFrame([data.dict()])
    probabilities = model.predict_proba(input_df)[0]

    top3_idx = np.argsort(probabilities)[-3:][::-1]
    crops = encoder.inverse_transform(top3_idx)
    confidence = probabilities[top3_idx]

    results = [
        {"crop": crops[i], "confidence": round(float(confidence[i]), 3)}
        for i in range(3)
    ]

    return {
        "top_3_recommendations": results
    }
