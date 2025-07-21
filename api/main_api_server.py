from fastapi import FastAPI
from api.routes import multi_region, primary_outbreak, personal_risk, contact_risk, explainability_model, symptom_standardizer
import logging
import os

app = FastAPI(
    title="AI Health Risk Prediction APIs",
    description="Includes outbreak prediction, personal risk scoring, and contact evaluation.",
    version="1.0.0"
)

app.include_router(multi_region.router, prefix="/predict/multi-region", tags=["Multi-Region Outbreak"])
app.include_router(primary_outbreak.router, prefix="/predict/primary_outbreak", tags=["Primary-Outbreak Outbreak"])
app.include_router(personal_risk.router, prefix="/predict/personal-risk", tags=["Personal Risk"])
app.include_router(contact_risk.router, prefix="/predict/contact-risk",tags=["Contact Risk"])
app.include_router(explainability_model.router, prefix="/risk/explainability_model",tags=["Risk Explainability"])
app.include_router(symptom_standardizer.router, prefix="/risk/symptom_standardizer", tags=["Symptom Standardizer"])





# Ensure logs directory exists
os.makedirs("Pandemic-intelligence---response-OS/logs", exist_ok=True)

# Configure logging to write to file
logging.basicConfig(
    filename="logs/prediction_logs.log",     # Log file path
    filemode="a",                             # Append mode
    level=logging.INFO,                       # Log level (can be DEBUG for more detail)
    format="%(asctime)s - %(levelname)s - %(message)s"
)