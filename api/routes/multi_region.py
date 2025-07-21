from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from api.utils.loader import load_multi_region_model
from api.utils.feature_extraction import preprocess_multi_region_input

# Setup logger
logger = logging.getLogger("multi_region_outbreak_logger")
logger.setLevel(logging.INFO)
log_file = "logs/multi_region_outbreak.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load model
model = load_multi_region_model()

router = APIRouter()

class OutbreakInput(BaseModel):
    population: int
    avg_daily_mobility: float
    avg_contact_rate: float
    R0_estimate: float
    policy_effectiveness_score: float
    transmission_route_from_enc: int

@router.post("/")
def predict_outbreak(input_data: OutbreakInput):
    try:
        logger.info("Incoming Request Data: %s", input_data.dict())

        df_input = pd.DataFrame([input_data.dict()])
        processed = preprocess_multi_region_input(df_input)

        logger.info("Processed Input Data: %s", processed.to_dict(orient="records"))

        prediction = model.predict(processed)

        spread_prob = float(prediction[0][0]) if prediction.ndim == 2 else float(prediction[0])
        eta_days = float(prediction[0][1]) if prediction.ndim == 2 and prediction.shape[1] > 1 else None

        explanation = {
            "R0_estimate": f"High R₀ ({input_data.R0_estimate}) means fast transmission.",
            "population": f"Large population ({input_data.population:,}) leads to high exposure risk.",
            "avg_contact_rate": f"Contact rate of {input_data.avg_contact_rate:.2f} increases spread.",
            "avg_daily_mobility": f"Mobility {input_data.avg_daily_mobility:.2f} increases inter-regional risk.",
            "policy_effectiveness_score": f"Policy score of {input_data.policy_effectiveness_score} reflects mitigation efforts.",
            "transmission_route_from_enc": f"Route encoding {input_data.transmission_route_from_enc} denotes spread channel."
        }

        result = {
            "predicted_spread_probability": round(spread_prob, 4),
            "input_summary": input_data.dict(),
            "explanation": explanation
        }

        if eta_days is not None:
            result["predicted_eta_days"] = round(eta_days, 2)
            result["Predicted Spread Prob"] = round(spread_prob, 4)

        logger.info("Prediction Result: %s", result)

        return result

    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        return {"error": "Prediction failed", "details": str(e)}
