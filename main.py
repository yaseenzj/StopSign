from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the API
app = FastAPI(title="FisherSafe AI Backend")

# CRITICAL FOR HACKATHONS: Allow CORS so your HTML map can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for local testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load your trained Random Forest Model
model = joblib.load('marine_risk_model.pkl')

# 2. Define what the frontend should send us
class WeatherData(BaseModel):
    wind_speed: float
    wave_height: float
    precipitation: float

# 3. Create the Risk Engine Endpoint
@app.post("/api/v1/risk-check")
async def calculate_risk(data: WeatherData):
    # Convert input to the format the model expects
    input_df = pd.DataFrame([data.model_dump()])
    
    # Get the prediction (0=Green, 1=Yellow, 2=Red)
    prediction = model.predict(input_df)[0]
    
    # Get the confidence percentage (e.g., 85% sure it's RED)
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities) * 100
    
    # Map the number to a readable status
    risk_mapping = {0: "GREEN", 1: "YELLOW", 2: "RED"}
    risk_level = risk_mapping[prediction]
    
    return {
        "status": "success",
        "risk_level": risk_level,
        "confidence": round(confidence, 2),
        "message": f"AI predicted {risk_level} with {round(confidence, 2)}% certainty."
    }

# Run this from the terminal using: uvicorn main:app --reload