from fastapi import FastAPI
from pydantic import BaseModel
import joblib


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(
    title="Email Spam Detection API",
    version="1.0.0"
)

# Request body
class Email(BaseModel):
    message: str

@app.get("/")
def home():
    return {
        "message": "Email Spam Detection API is running!"
    }

@app.post("/predict")
def predict(email: Email):

    
    transformed = vectorizer.transform([email.message])

    prediction = model.predict(transformed)[0]

    if prediction == 1:
        result = "Ham"
    else:
        result = "Spam"

    return {
        "message": email.message,
        "prediction": result
    }

#python -m uvicorn main:app --reload