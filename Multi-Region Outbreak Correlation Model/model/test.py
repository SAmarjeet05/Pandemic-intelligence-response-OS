# test_outbreak_model.py

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from pathlib import Path
import numpy as np

# === Load Processed Data and Model ===
base_path = Path(__file__).resolve().parent.parent 
data_path = base_path / "data" / "raw"
model_path = base_path / "model" / "multi_region_model.pkl"
processed_data_path = base_path / "data" / "processed" / "final_dataset_with_labels.csv"

df = pd.read_csv(processed_data_path)
model = joblib.load(model_path)

# === Prepare Features and Labels ===
features = ['log_population', 'avg_daily_mobility', 'avg_contact_rate', 'R0_estimate',
            'policy_effectiveness_score', 'transmission_route_from_enc']
targets = ['spread_prob', 'eta_days']

X = df[features]
y_true = df[targets]

# === Make Predictions ===
y_pred = model.predict(X)

# === Evaluate ===
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print("=== Model Evaluation ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# === Predict on Sample Input ===
sample_input = {
    'log_population': np.log1p(20000000),
    'avg_daily_mobility': 45000,
    'avg_contact_rate': 43494.0403,
    'R0_estimate': 1.5,
    'policy_effectiveness_score': 4.5,
    'transmission_route_from_enc': 1
}
import numpy as np
sample_df = pd.DataFrame([sample_input])
sample_pred = model.predict(sample_df)

print("\n=== Sample Prediction ===")
print("Input:", sample_input)
print(f"Predicted Spread Prob: {round(sample_pred[0][0], 4)}")
print(f"Predicted ETA Days: {round(sample_pred[0][1], 2)}")
