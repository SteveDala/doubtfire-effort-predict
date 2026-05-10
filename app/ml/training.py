import xgboost as xgb
import numpy as np
import logging

logger = logging.getLogger(__name__)


def train_dummy_model():
    logger.info("Training dummy XGBoost model...")

    X, y = generate_dummy_dataset()
    dtrain = xgb.DMatrix(X, label=y)

    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse"
        },
        dtrain=dtrain,
        num_boost_round=50
    )

    logger.info("Training complete")

    return model


def generate_dummy_dataset(n=500, seed=42):
    np.random.seed(seed)

    X = np.zeros((n, 3), dtype=float)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        hours = np.random.randint(1, 40)
        grade = np.random.randint(0, 3)
        days = np.random.randint(1, 45)

        effort = (
            1.5 * hours +
            1.2 * grade +
            0.95 * days +
            np.random.normal(0, 3)
        )

        X[i] = [hours, grade, days]
        y[i] = effort

    return X, y
