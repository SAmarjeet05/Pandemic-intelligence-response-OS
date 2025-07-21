import joblib
import os

def load_multi_region_model():
    model_path = os.path.join("..","Pandemic intelligence & response OS", "Multi-Region Outbreak Correlation Model", "model", "multi_region_model.pkl")
    return joblib.load(model_path)



def load_primary_outbreak_model():
    model_path = os.path.join("..","Pandemic intelligence & response OS","Primary Outbreak Risk Predictor", "model", "auto_trained_model.pkl")
    return joblib.load(model_path)


def load_primary_outbreak_scaler():
    scaler_path = os.path.jio("..","Pandemic intelligence & response OS","Primary Outbreak Risk Predictor", "model", "feature_scaler.pkl")
    return joblib.load(scaler_path)


# api/utils/loader.py

def build_risk_prompt(data: dict) -> str:
    prompt = (
        f"Assess the health risk for the following individual:\n"
        f"Age: {data['age']}\n"
        f"Gender: {data['gender']}\n"
        f"Location: {data['location']}\n"
        f"Pre-existing conditions: {', '.join(data['pre_existing_conditions'])}\n"
        f"Symptoms: {', '.join(data['symptoms'])}\n"
        f"Lifestyle factors: {', '.join(data['lifestyle_factors'])}\n"
        f"Please return a brief medical risk assessment."
    )
    return prompt



def build_contact_risk_prompt(data: dict) -> str:
    return f"""
You are an expert epidemiologist. Assess the COVID-19 risk from the following contact encounter:

- Contact Duration: {data['contact_duration_minutes']} minutes
- Location Type: {data['location_type']}
- Mask Used: {"Yes" if data['mask_used'] else "No"}
- Social Distancing Followed: {"Yes" if data['social_distancing_followed'] else "No"}
- Contact’s Symptoms: {', '.join(data['contact_symptoms']) if data['contact_symptoms'] else "None"}
- Contact Travel History: {"Yes" if data['contact_travel_history'] else "No"}
- User’s Vaccination Status: {data['vaccination_status']}
- User’s Health Conditions: {', '.join(data['user_health_conditions']) if data['user_health_conditions'] else "None"}

Based on these details, provide a risk assessment on a scale of LOW, MODERATE, HIGH. Also explain the reasoning.
"""




# utils/build_explanation_prompt.py

def build_explanation_prompt(data: dict) -> str:
    return (
        f"Evaluate the following patient profile and explain, step by step, the reasoning used to assess health risk:\n\n"
        f"Age: {data['age']}\n"
        f"Gender: {data['gender']}\n"
        f"Location: {data['location']}\n"
        f"Pre-existing conditions: {', '.join(data['pre_existing_conditions'])}\n"
        f"Symptoms: {', '.join(data['symptoms'])}\n"
        f"Lifestyle factors: {', '.join(data['lifestyle_factors'])}\n\n"
        f"Step-by-step reasoning:\n"
    )


# utils/build_standardization_prompt.py

def build_standardization_prompt(symptom_description: str) -> str:
    return (
        f"You are a medical assistant.\n"
        f"Standardize the following symptom description into a list of clinical terms:\n\n"
        f"Patient says: \"{symptom_description}\"\n\n"
        f"Standardized Symptoms (comma-separated):"
    )


