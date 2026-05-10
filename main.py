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
    model.load_model("model.json") # load trained model file
else:
    print("model.json not found")

# Request payload
class Task(BaseModel):
    estimated_hours: float
    target_grade: str
    start_date: str
    due_date: str

# Encode grades
def encode_grade(grade: str):
    mapping = {"HD": 4, "D": 3, "C": 2, "P": 1, "N": 0}
    return mapping.get(grade.upper(), 2)

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

    # Build feature vector
    features = np.array([
        task.estimated_hours,
        encode_grade(task.target_grade),
        days_avail
    ]).reshape(1, -1)

    # Predict
    #prediction = model.predict(features)[0]

    #return {"predicted_effort": float(prediction)}

    # Return estimated_hours as of now
    return {"predicted_effort": float(task.estimated_hours)}


@app.post("/train")
def train(payload: dict):
    logger.info(f"Received training payload: {payload.keys()}")
    return {"training_results": "You have hit the endpoint"}
