from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import os
from api.utils.loader import load_primary_outbreak_model, load_primary_outbreak_scaler

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    filename="logs/prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the model and scaler
model = load_primary_outbreak_model()

try:
    scaler = load_primary_outbreak_scaler()
except Exception as e:
    scaler = None
    logging.warning(f"Scaler not loaded: {e}")

router = APIRouter()

# Define input schema
class PrimaryOutbreakInput(BaseModel):
    cluster_score: float
    mobility_index: float
    residential_change: float
    transport_usage: float
    icu_usage_per_1000: float
    tweet_volume: float
    sentiment_score: float
    temp_avg: float
    humidity: float
    air_quality_index: float
    rainfall_mm: float
    keyword_0: int = 0
    keyword_1: int = 0
    keyword_2: int = 0
    keyword_3: int = 0
    keyword_4: int = 0

@router.post("/")
def predict_primary_outbreak(input_data: PrimaryOutbreakInput):
    try:
        # Log raw input
        input_dict = input_data.dict()
        logging.info(f"Incoming raw input: {input_dict}")

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Log DataFrame before scaling
        logging.info(f"Raw features before scaling:\n{input_df}")

        # Scale if scaler is used
        if scaler:
            input_df_scaled = scaler.transform(input_df)
            logging.info(f"Scaled features:\n{input_df_scaled}")
        else:
            input_df_scaled = input_df
            logging.info(f"No scaling applied. Using raw features.")

        # Model prediction
        prob = model.predict_proba(input_df_scaled)[0][1]
        prediction = int(prob >= 0.29)
        message = "Outbreak Likely" if prediction == 1 else "No Outbreak"

        # Log prediction results
        logging.info(f"Prediction: {prediction}, Probability: {round(prob, 4)}, Message: {message}")

        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "message": message
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
