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


# logger = logging.getLogger("uvicorn.error")
# logger.info(f"XGBoost version: {xgb.__version__}")
#
# MODEL_PATH = os.getenv("MODEL_PATH", "model.json")
#
# model = None
# model_loaded = False
#
#
# def load_or_train_model():
#     global model, model_loaded
#
#     try:
#         if os.path.exists(MODEL_PATH):
#             logger.info(f"Loading model from {MODEL_PATH}")
#
#             model = xgb.Booster()
#             model.load_model(MODEL_PATH)
#
#         else:
#             logger.warning("Model not found - training new model")
#
#             model = train_model()
#             model.save_model(MODEL_PATH)
#
#         if model is None or not hasattr(model, "predict"):
#             raise RuntimeError("Loaded model is invalid after load/train")
#
#         model_loaded = True
#         logger.info("XGBoost model ready for inference")
#
#     except Exception as e:
#         model = None
#         model_loaded = False
#         logger.error(f"Failed to load model: {e}")
#
#
# def train_model():
#     logger.info("Training XGBoost model...")
#
#     # Dummy data in trining function for now
#     X = np.array([
#         [10, 85, 5],
#         [5, 70, 2],
#         [20, 90, 10]
#     ])
#
#     # also dummy predicted efforts for now
#     y = np.array([40, 20, 80])
#
#     dtrain = xgb.DMatrix(X, label=y)
#
#     model = xgb.train(
#         params={
#             "objective": "reg:squarederror",
#             "eval_metric": "rmse"
#         },
#         dtrain=dtrain,
#         num_boost_round=50
#     )
#
#     logger.info("Training complete")
#
#     return model
#
#
# def days_between(start: str, end: str):
#     """
#     Turns two dates in ISO format to datetimes. Then, calculates the difference
#     and stores the result as days. The number of days given to complete a task
#     will factor into the predicted effort.
#     """
#     s = datetime.fromisoformat(start)
#     e = datetime.fromisoformat(end)
#     return (e - s).days
#
# # Prediction endpoint
#
#
# @app.post(
#     "/predict",
#     tags=["Prediction"],
#     summary="Predicts effort for a given Task Definition",
#     description="Uses a trained XGBoost model to estimate effort based on" +
#                 " Task Definition information.",
#     response_description="Raw predicted effort score"
# )
# def predict_effort(task: TaskDefinition):
#     logger.info(f"Received task payload: {task.dict()}")
#
#     if not model_loaded or model is None:
#         raise HTTPException(
#             status_code=503,
#             detail="Model not available (not trained or loaded)"
#         )
#
#     # Convert dates
#     days_avail = days_between(task.start_date, task.due_date)
#
#     # Build feature vector
#     features = np.array([
#         task.estimated_hours,
#         task.target_grade,
#         days_avail
#     ]).reshape(1, -1)
#
#     # Predict
#     dtest = xgb.DMatrix(features)
#     prediction = model.predict(dtest)[0]
#
#     return {"predicted_effort": float(prediction)}
#
#     # DEBUG: Return estimated_hours instead
#     # return {"predicted_effort": float(task.estimated_hours)}
#
#
# @app.post("/train")
# def train(payload: dict):
#     logger.info(f"Received training payload: {payload.keys()}")
#     return {"training_results": "You have hit the endpoint"}
#
