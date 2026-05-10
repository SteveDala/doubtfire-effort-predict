import os
import logging
import xgboost as xgb

from app.ml.training import train_dummy_model

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.json")

model = None
model_loaded = False


def load_or_train_model():
    global model, model_loaded

    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")

            model = xgb.Booster()
            model.load_model(MODEL_PATH)

        else:
            logger.warning("Model not found - training dummy model")

            model = train_dummy_model()
            model.save_model(MODEL_PATH)

        model_loaded = True

        logger.info("Model ready for inference")

    except Exception as e:
        model = None
        model_loaded = False
        logger.error(f"Failed to initialize model: {e}")


def get_model():
    return model
