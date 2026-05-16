from fastapi import FastAPI
from app.api.routes.predict import router as predict_router
# from app.api.routes.train import router as train_router

from app.services.model_service import load_or_train_model
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="OnTrack Task Effort Prediction API",
    version="0.2.0"
)


@app.on_event("startup")
def startup_event():
    load_or_train_model()


app.include_router(predict_router)
# app.include_router(train_router)
