import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# -------------------------------
# Load Raw Datasets
# -------------------------------
base_path = Path(__file__).resolve().parent.parent 
data_path = base_path / "data" / "raw"
model_path = base_path / "model"
processed_data_path = base_path / "data" / "processed"



epi = pd.read_csv(data_path / "epidemiological_data.csv")
mobility = pd.read_csv(data_path / "mobility_data.csv", parse_dates=["date"])
policy = pd.read_csv(data_path / "policy_data.csv")
node_features = pd.read_csv(data_path / "node_features.csv")

# --- avg_daily_mobility ---
mobility_inflow = mobility.groupby('to_region')['daily_travelers'].mean().reset_index()
mobility_inflow.columns = ['region', 'avg_daily_mobility']

# --- avg_contact_rate (simplified as inflow/population ratio) ---
features_df = pd.merge(node_features, mobility_inflow, on='region', how='left')
features_df['avg_daily_mobility'] = features_df['avg_daily_mobility'].fillna(0)
features_df['avg_contact_rate'] = features_df['avg_daily_mobility'] / (features_df['population'] + 1)

# --- outbreak_status: recent cases > threshold ---
epi_recent = epi.sort_values('date').groupby('region').tail(7)
recent_cases = epi_recent.groupby('region')['daily_cases'].mean().reset_index()
recent_cases.columns = ['region', 'avg_recent_cases']
features_df = pd.merge(features_df, recent_cases, on='region', how='left')
features_df['outbreak_status'] = features_df['avg_recent_cases'].apply(lambda x: 1 if x > 50 else 0)

# --- transmission_route_from: most common source of inflow ---
most_common_source = mobility.groupby('to_region')['from_region'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown').reset_index()
most_common_source.columns = ['region', 'transmission_route_from']
features_df = pd.merge(features_df, most_common_source, on='region', how='left')
le_route = LabelEncoder()
features_df['transmission_route_from_enc'] = le_route.fit_transform(features_df['transmission_route_from'].astype(str))

# --- eta_days: heuristic from R0_estimate and contact rate ---
epi_latest = epi.sort_values('date').groupby('region').tail(1)[['region', 'R0_estimate']]
features_df = pd.merge(features_df, epi_latest, on='region', how='left')
features_df['eta_days'] = (10 / (features_df['R0_estimate'] + 0.01)).clip(1, 20)

# --- policy_effectiveness_score ---
latest_policy = policy.sort_values('date').groupby('region').tail(1)[['region', 'policy_type']]
features_df = pd.merge(features_df, latest_policy, on='region', how='left')
features_df['policy_effectiveness_score'] = features_df['policy_type'].apply(
    lambda x: 1.0 if x in ['lockdown', 'travel_ban'] else (0.7 if x in ['mask_mandate'] else 0.5))


# Normalize each feature
norm = lambda col: (col - col.min()) / (col.max() - col.min() + 1e-6)

R0 = norm(features_df['R0_estimate'])
POP = norm(features_df['population'])
CR = norm(features_df['avg_contact_rate'])
MOB = norm(features_df['avg_daily_mobility'])
ROUTE = norm(features_df['transmission_route_from_enc'])
POL = norm(features_df['policy_effectiveness_score'])

# Weighted combination based on importance
spread_raw = (
    0.30 * R0 +
    0.20 * POP +
    0.18 * CR +
    0.15 * MOB +
    0.10 * ROUTE +
    0.07 * (1 - POL)  # higher policy effectiveness = lower spread
)

# Normalize final score and clip
features_df['spread_prob'] = (0.01 + 0.89 * spread_raw).clip(0, 1)



features_df['log_population'] = np.log1p(features_df['population'])


# === Final Dataset ===
final_columns = ['log_population', 'avg_daily_mobility', 'avg_contact_rate',
                 'R0_estimate', 'policy_effectiveness_score', 'transmission_route_from_enc']

targets = ['spread_prob', 'eta_days']

final_dataset = features_df.dropna(subset=final_columns + targets)

# Save processed dataset
os.makedirs(base_path / "data" / "processed", exist_ok=True)
final_dataset.to_csv(processed_data_path /'final_dataset_with_labels.csv', index=False)

# === Train Model ===
X = final_dataset[final_columns]
y = final_dataset[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, max_depth=10 , random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Save Model
os.makedirs(model_path, exist_ok=True)
joblib.dump(model, model_path / 'multi_region_model.pkl')

print("Model training completed and saved successfully.")

import matplotlib.pyplot as plt

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=True).plot(kind='barh', figsize=(10,6))
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

