from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import numpy as np
from joblib import load
import tensorflow as tf
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = load(os.path.join(BASE_DIR, 'scaler_diabetes.joblib'))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model_diabetes.h5'))

class PredictionRequest(BaseModel):
    inputs: Dict[str, Any]

@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = request.inputs

    try:
        # Define feature order (must match training order)
        feature_order = ['bmi', 'age', 'genhlth', 'income', 'highBP', 'education', 'physhlth']
        
        # Extract and validate features
        features = []
        for f in feature_order:
            val = inputs.get(f)
            if val is None:
                raise HTTPException(status_code=400, detail=f"Missing feature: {f}")
            try:
                features.append(float(val))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid value for {f}: must be numeric")

        input_array = np.array([features])
        print("Raw input features:", input_array)

        # Scale the features (CRITICAL STEP)
        scaled_input = scaler.transform(input_array)
        print("Scaled features:", scaled_input)

        # Make prediction
        prediction = model.predict(scaled_input)
        diabetes_score = float(prediction[0][0])

        return {
            "predictions": [
                {"label": "non-diabetic", "score": round(1 - diabetes_score, 4)},
                {"label": "diabetic", "score": round(diabetes_score, 4)}
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
