from fastapi import APIRouter, HTTPException
from app.ml.prediction import predict_raw_task_effort
from app.schemas.responses import PredictionResponse
from app.schemas.task_definition import TaskDefinition


router = APIRouter()


@router.post(
    "/predict",
    tags=["Prediction"],
    summary="Predicts effort for a given Task Definition",
    description="Uses a trained XGBoost model to estimate effort based on" +
                " Task Definition information.",
    response_description="Raw predicted effort score",
    response_model=PredictionResponse
)
def predict_effort(task: TaskDefinition):

    try:
        prediction = predict_raw_task_effort(task)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model unavailable")

    return {"predicted_effort": float(prediction)}
