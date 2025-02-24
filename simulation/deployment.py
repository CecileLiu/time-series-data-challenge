from fastapi import FastAPI, HTTPException
import uvicorn
import torch
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load trained models
sarima_model = joblib.load("sarima_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
transformer_model = torch.load("transformer_model.pth")
transformer_model.eval()

# Define request model
class PredictionRequest(BaseModel):
    features: list
    model_type: str  # "sarima", "xgboost", or "transformer"

# Prediction function
def predict(model, input_data, model_type):
    if model_type == "sarima":
        return model.forecast(steps=1)[0]
    elif model_type == "xgboost":
        return model.predict(np.array(input_data).reshape(1, -1))[0]
    elif model_type == "transformer":
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            return model(input_tensor).item()
    else:
        raise ValueError("Invalid model type")

@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    try:
        prediction = predict(
            model=sarima_model if request.model_type == "sarima" else xgb_model if request.model_type == "xgboost" else transformer_model,
            input_data=request.features,
            model_type=request.model_type
        )
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
