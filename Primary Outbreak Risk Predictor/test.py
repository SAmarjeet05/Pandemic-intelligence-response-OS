import joblib
import pandas as pd
import numpy as np

# Load your scaler
scaler = joblib.load("model/feature_scaler.pkl")

# Known scaled test row (excluding categorical like date, district_id, keywords)
scaled_values = [[
    0.1652765464529907,   # mobility_index
    -0.44085231129503544, # residential_change
    -0.5283393027835652,  # transport_usage
    -0.8196549343136561,  # icu_usage_per_1000
    0.397422912917571,    # tweet_volume
    -1.4702346751630766,  # sentiment_score
    -1.6109808822802074,  # temp_avg
    -1.349650653401665,   # humidity
    -1.567732208505871,   # air_quality_index
    -1.3099954977327315,  # rainfall_mm
    0.0                   # cluster_score
]]

# Inverse transform
raw_values = scaler.inverse_transform(scaled_values)

# Show unscaled version
columns = [
    "mobility_index", "residential_change", "transport_usage",
    "icu_usage_per_1000", "tweet_volume", "sentiment_score",
    "temp_avg", "humidity", "air_quality_index", "rainfall_mm",
    "cluster_score"
]

unscaled_row = pd.DataFrame(raw_values, columns=columns)
print(unscaled_row.round(3).to_dict(orient="records")[0])
