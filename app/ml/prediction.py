from app.ml.features import build_features
from app.services.model_service import get_model
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


def predict_raw_task_effort(task):

    model = get_model()

    features = build_features(task)
    dtest = xgb.DMatrix(features)
    logger.info(f"New features received: {features[0]}")

    prediction = model.predict(dtest)[0]
    logger.info(f"Predicted effort for task: {prediction}")

    return prediction
