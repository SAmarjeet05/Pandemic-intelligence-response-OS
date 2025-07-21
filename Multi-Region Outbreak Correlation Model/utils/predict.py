# predict.py

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path

# === Paths ===
base_path = Path(__file__).resolve().parent.parent 
data_path = base_path / "data" / "raw"
model_path = base_path / "model" / "multi_region_model.pkl"
processed_data_path = base_path / "data" / "processed" / "final_dataset_with_labels.csv"

# === Load model and dataset ===
model = joblib.load(model_path)
data = pd.read_csv(processed_data_path)

# === Define features and targets ===
feature_cols = ['population', 'avg_daily_mobility', 'avg_contact_rate', 'R0_estimate',
                'policy_effectiveness_score', 'transmission_route_from_enc']
target_cols = ['spread_prob', 'eta_days']

# === Split into X and y ===
X = data[feature_cols]
y_true = data[target_cols]

# === Make Predictions ===
y_pred = model.predict(X)

# === Evaluate Model ===
mse = mean_squared_error(y_true, y_pred)
print(f"Prediction MSE on full processed dataset: {mse:.4f}")

# === Example: Predict for a single region ===
sample = X.iloc[0:1]
prediction = model.predict(sample)
print("\n=== Sample Prediction ===")
print(f"Input: {sample.to_dict(orient='records')[0]}")
print(f"Predicted Spread Prob: {prediction[0][0]:.4f}")
print(f"Predicted ETA Days: {prediction[0][1]:.2f}")
