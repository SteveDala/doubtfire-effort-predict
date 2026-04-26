from fastapi import FastAPI
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    logger.info(f"Received payload: {payload}")
    return {"predicted_effort": 42}
