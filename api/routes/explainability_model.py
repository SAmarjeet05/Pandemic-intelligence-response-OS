from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import logging
from api.utils.loader import build_explanation_prompt

router = APIRouter()

logger = logging.getLogger("explainability_model_logger")
logger.setLevel(logging.INFO)
log_file = "logs/explainability_model.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ExplainInput(BaseModel):
    age: int
    gender: str
    location: str
    pre_existing_conditions: List[str]
    symptoms: List[str]
    lifestyle_factors: List[str]

def run_cot_explanation(prompt: str) -> str:
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
        logger.error("Ollama CoT error: %s", str(e), exc_info=True)
        raise

@router.post("/explain/")
async def explain_decision(input_data: ExplainInput):
    try:
        logger.info("Generating explainability for: %s", input_data.dict())
        prompt = build_explanation_prompt(input_data.dict())
        explanation = run_cot_explanation(prompt)
        logger.info("Generated explanation: %s", explanation)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
