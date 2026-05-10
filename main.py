from fastapi import FastAPI
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    logger.info(f"Received prediction payload: {payload}")
    # predicted_effort = model.predict(processed_payload)
    return {"predicted_effort": 42}

@app.post("/train")
def train(payload: dict):
    logger.info(f"Received training payload: {payload.keys()}")
    return {"training_results": "You have hit the endpoint"}
