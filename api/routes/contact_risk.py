from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import requests
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
from concurrent.futures import ThreadPoolExecutor
from api.utils.loader import build_contact_risk_prompt


logger = logging.getLogger("contact_risk_evaluation_logger")
logger.setLevel(logging.INFO)
log_file = "logs/contact_risk.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()
executor = ThreadPoolExecutor()

class ContactRiskInput(BaseModel):
    contact_duration_minutes: int
    location_type: str
    mask_used: bool
    social_distancing_followed: bool
    contact_symptoms: List[str]
    contact_travel_history: bool
    vaccination_status: str
    user_health_conditions: List[str]



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
    

@router.post("/")
async def assess_risk(input_data: ContactRiskInput):
    try:
        logger.info("Received personal risk request: %s", input_data.dict())
        prompt = build_contact_risk_prompt(input_data.dict())
        logger.info("Prompt constructed: %s", prompt.replace('\n', ' | '))

        response_text = run_mistral(prompt)

        logger.info("Ollama response: %s", response_text.replace('\n', ' | '))

        return {"assessment": response_text}

    except Exception as e:
        logger.error("Error in personal risk assessment: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Model inference failed")
