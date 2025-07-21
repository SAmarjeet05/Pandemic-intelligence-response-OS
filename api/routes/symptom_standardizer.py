from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import logging
from api.utils.loader import build_standardization_prompt

router = APIRouter()

logger = logging.getLogger("symptom_standardizer_logger")
logger.setLevel(logging.INFO)
log_file = "logs/symptom_standardizer.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class SymptomInput(BaseModel):
    symptom_description: str

def run_standardizer_llm(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        logger.error("Symptom LLM error: %s", str(e), exc_info=True)
        raise

@router.post("/standardize/")
async def standardize_symptoms(input_data: SymptomInput):
    try:
        logger.info("Standardizing symptoms for input: %s", input_data.symptom_description)
        prompt = build_standardization_prompt(input_data.symptom_description)
        standardized = run_standardizer_llm(prompt)
        logger.info("Standardized output: %s", standardized)
        return {"standardized_symptoms": standardized.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
