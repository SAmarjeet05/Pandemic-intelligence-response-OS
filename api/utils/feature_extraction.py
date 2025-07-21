# api/utils/preprocessing.py

import pandas as pd
import numpy as np

# --------------------------
# Multi-Region Outbreak Correlation Model Preprocessing
# --------------------------

def preprocess_multi_region_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input DataFrame for the Multi-Region Outbreak Correlation model.
    Returns a single-row DataFrame in the correct format.
    """
    # Feature engineering as per model training
    df['log_population'] = np.log1p(df['population'])

    # Final feature set (must match training)
    features = df[[
        'log_population',
        'avg_daily_mobility',
        'avg_contact_rate',
        'R0_estimate',
        'policy_effectiveness_score',
        'transmission_route_from_enc'
    ]]

    return features



# --------------------------
# Primary Outbreak Detection Model Preprocessing
# --------------------------

def preprocess_primary_outbreak_input(data: dict) -> np.ndarray:
    """
    Converts input data dict into a processed numpy array for the Primary Outbreak model.
    Mimics the same format used during model training.
    """
    # Required order of features during training
    features_order = [
        "cluster_score",
        "mobility_index",
        "residential_change",
        "transport_usage",
        "icu_usage_per_1000",
        "tweet_volume",
        "sentiment_score",
        "temp_avg",
        "humidity",
        "air_quality_index",
        "rainfall_mm",
        "keyword_0",
        "keyword_1",
        "keyword_2",
        "keyword_3",
        "keyword_4"
    ]

    try:
        input_vector = [data[feature] for feature in features_order]
        input_array = np.array(input_vector).reshape(1, -1)
    except KeyError as e:
        raise ValueError(f"Missing feature in input: {e}")

    return input_array
