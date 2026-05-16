from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

import xgboost as xgb
import numpy as np
import os
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# Load XGBoost model
model = None
if os.path.exists("model.json"):
    model = xgb.XGBRegressor()
    model.load_model("model.json")
else:
    print("model.json not found")

# Request payload
class Task(BaseModel):
    estimated_hours: float
    target_grade: int   # now passed as integer (HD=3, D=2, C=1, P=0)
    start_date: str
    due_date: str

# Conversion
def days_between(start: str, end: str):
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    return (e - s).days

# Prediction endpoint
@app.post("/predict")
def predict_effort(task: Task):
    logger.info(f"Received task payload: {task.dict()}")

    # Convert dates
    days_avail = days_between(task.start_date, task.due_date)

    # Build feature vector (no encode_grade needed)
    features = np.array([
        task.estimated_hours,
        task.target_grade,
        days_avail
    ]).reshape(1, -1)

    # If model exists, use it; otherwise return dummy
    if model:
        prediction = model.predict(features)[0]
        return {"predicted_effort": float(prediction)}

    return {"predicted_effort": float(task.estimated_hours)}

# Training endpoint (from Steven)
@app.post("/train")
def train(payload: dict):
    logger.info(f"Received training payload: {payload.keys()}")
    return {"training_results": "You have hit the endpoint"}
