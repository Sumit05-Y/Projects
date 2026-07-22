# main.py
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Breast Cancer Classifier API")

model = joblib.load("breast_cancer_svm.pkl")
scaler = joblib.load("breast_cancer_scaler.pkl")

class PatientFeatures(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)

@app.get("/")
def root():
    return {"message": "Breast Cancer Classifier API", "status": "running"}

@app.post("/predict")
def predict(data: PatientFeatures):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    label = "malignant" if prediction == 0 else "benign"
    return {
        "prediction": label,
        "confidence": float(max(probability)),
        "probabilities": {
            "malignant": float(probability[0]),
            "benign": float(probability[1])
        }
    }