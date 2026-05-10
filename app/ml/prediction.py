from app.ml.features import build_features
from app.services.model_service import get_model
import xgboost as xgb


def predict_raw_task_effort(task):

    model = get_model()

    features = build_features(task)
    dtest = xgb.DMatrix(features)
    print(features)
    prediction = model.predict(dtest)[0]

    return prediction
