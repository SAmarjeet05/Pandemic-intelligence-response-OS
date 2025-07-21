import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from api.utils.loader import build_risk_prompt  # Prompt builder function

# ------------------ Setup ------------------
logger = logging.getLogger("personal_risk_assessment_logger")
logger.setLevel(logging.INFO)
log_file = "logs/personal_risk.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

router = APIRouter()
executor = ThreadPoolExecutor()


# ------------------ Request Schema ------------------
class RiskInput(BaseModel):
    age: int
    gender: str
    location: str
    pre_existing_conditions: List[str]
    symptoms: List[str]
    lifestyle_factors: List[str]

# ------------------ Inference Function ------------------

def run_mistral(prompt: str) -> str:
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


# ------------------ Endpoint ------------------
@router.post("/")
async def assess_risk(input_data: RiskInput):
    try:
        logger.info("Received personal risk request: %s", input_data.dict())
        prompt = build_risk_prompt(input_data.dict())
        logger.info("Prompt constructed: %s", prompt.replace('\n', ' | '))

        response_text = run_mistral(prompt)

        logger.info("Ollama response: %s", response_text.replace('\n', ' | '))

        return {"assessment": response_text}

    except Exception as e:
        logger.error("Error in personal risk assessment: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Model inference failed")
