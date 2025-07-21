# feature_extraction.py
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os

# ✅ Added for SMOTE
from imblearn.over_sampling import SMOTE

## Load datasets
base_path = Path(__file__).resolve().parent.parent 
data_path = base_path / "data" / "raw"
model_path = base_path / "model"
processed_data_path = base_path / "data" / "processed"

symptoms_df = pd.read_csv(data_path / "symptomReport.csv", parse_dates=['timestamp'])
mobility_df = pd.read_csv(data_path / "mobilityData.csv")
residential_df = pd.read_csv(data_path / "residentialMovement.csv")
transport_df = pd.read_csv(data_path / "publicTransport.csv")
healthcare_df = pd.read_csv(data_path / "healthcareUsage.csv")
outbreak_labels_df = pd.read_csv(data_path / "outbreakLabels.csv")
population_df = pd.read_csv(data_path / "populationData.csv")
sentiment_df = pd.read_csv(data_path / "socialMediaSentiment.csv")
environmental_df = pd.read_csv(data_path / "environmentalData.csv")

# 1. Cluster Score
symptoms_df['lat_bin'] = (symptoms_df['latitude'] * 100).astype(int)
symptoms_df['long_bin'] = (symptoms_df['longitude'] * 100).astype(int)
cluster_counts = symptoms_df.groupby(['lat_bin', 'long_bin']).size().reset_index(name='cluster_score')
avg_cluster_score = cluster_counts['cluster_score'].mean()

# 2. Mobility Index
mobility_df['mobility_index'] = (mobility_df['total_trips'] - mobility_df['baseline_trips']) / mobility_df['baseline_trips']

# 3. Public Transport Usage
transport_df['transport_usage'] = (transport_df['swipe_ins'] - transport_df['avg_swipe_baseline']) / transport_df['avg_swipe_baseline']

# 4. Residential Movement
residential_df['residential_change'] = (residential_df['avg_home_time_today'] - residential_df['avg_home_time_baseline']) / residential_df['avg_home_time_baseline']

# 5. Healthcare Usage
healthcare_df['date'] = pd.to_datetime(healthcare_df['date'])
healthcare_df['icu_usage_per_1000'] = healthcare_df['icu_beds_used'] / population_df['total_population'].mean() * 1000

# 6. Social Media Sentiment
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# 7. Environmental Data
environmental_df['date'] = pd.to_datetime(environmental_df['date'])

# Merge mobility, residential and transport
mobility_df['date'] = pd.to_datetime(mobility_df['date'])
residential_df['date'] = pd.to_datetime(residential_df['date'])
transport_df['date'] = pd.to_datetime(transport_df['date'])

mobility_df = mobility_df.merge(residential_df[['date', 'district_id', 'residential_change']], on=['date', 'district_id'], how='left')
mobility_df = mobility_df.merge(transport_df[['date', 'district_id', 'transport_usage']], on=['date', 'district_id'], how='left')

# Save processed mobility data
os.makedirs(base_path / "data/processed", exist_ok=True)
mobility_df.to_csv(processed_data_path / "mobilityDataProcessed.csv", index=False)

# Create features
features_df = mobility_df[['date', 'district_id', 'mobility_index', 'residential_change', 'transport_usage']]
features_df = features_df.merge(healthcare_df[['date', 'district_id', 'icu_usage_per_1000']], on=['date', 'district_id'], how='left')
features_df = features_df.merge(sentiment_df[['date', 'district_id', 'tweet_volume', 'sentiment_score', 'top_keywords']], on=['date', 'district_id'], how='left')
features_df = features_df.merge(environmental_df[['date','district_id', 'temp_avg', 'humidity', 'air_quality_index', 'rainfall_mm']],  on=['date', 'district_id'], how='left')

# Assign a constant average cluster score
features_df['cluster_score'] = avg_cluster_score

# Handle missing values
features_df.fillna(method='ffill', inplace=True)
features_df.fillna(method='bfill', inplace=True)

# Handle top_keywords (categorical) using CountVectorizer
vectorizer = CountVectorizer(max_features=5)
top_keywords_matrix = vectorizer.fit_transform(features_df['top_keywords'].fillna(""))
top_keyword_feature_names = [f"keyword_{i}" for i in range(top_keywords_matrix.shape[1])]
top_keywords_df = pd.DataFrame(top_keywords_matrix.toarray(), columns=top_keyword_feature_names)
features_df = pd.concat([features_df.reset_index(drop=True), top_keywords_df], axis=1)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['cluster_score', 'mobility_index', 'residential_change', 'transport_usage',
                      'icu_usage_per_1000', 'tweet_volume', 'sentiment_score', 'temp_avg',
                      'humidity', 'air_quality_index', 'rainfall_mm']
features_df[numerical_features] = scaler.fit_transform(features_df[numerical_features])

# Final features list
feature_cols = numerical_features + top_keyword_feature_names

# Save processed feature data
features_df.to_csv(processed_data_path / "feature_dataset.csv", index=False)
print("✅ Processed features saved to data/processed/feature_dataset.csv")

# ---------- Auto Training ----------
# Merge with labels
outbreak_labels_df['date'] = pd.to_datetime(outbreak_labels_df['date'])
features_df = features_df.merge(outbreak_labels_df[['date', 'district_id', 'outbreak']], on=['date', 'district_id'], how='left')
features_df.dropna(subset=['outbreak'], inplace=True)

# Define features and labels
X = features_df[feature_cols]
y = features_df['outbreak'].astype(int)

# ✅ Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("✅ After SMOTE resampling:\n", y_resampled.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Cross-validation accuracy
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"✅ Cross-validation accuracy: {cv_scores.mean() * 100:.2f}% (std: {cv_scores.std() * 100:.2f}%)")

# Save model and scaler
os.makedirs(base_path / "model", exist_ok=True)
joblib.dump(model, model_path / "auto_trained_model.pkl")
joblib.dump(scaler, model_path / "feature_scaler.pkl")

# Save final training dataset
features_df.to_csv(processed_data_path / "final_dataset_with_labels.csv", index=False)
print("✅ Final dataset saved to data/processed/final_dataset_with_labels.csv")

# Evaluate model
y_pred = model.predict(X_test)

# Save prediction report to file
report_text = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

with open(model_path / "classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report_text)
    f.write(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print("\n✅ Classification Report:\n")
print(report_text)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print("📁 Report saved to model/classification_report.txt")


from sklearn.metrics import precision_recall_curve

y_probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Show thresholds near 0.2 - 0.5
for t, p, r in zip(thresholds, precisions, recalls):
    if 0.2 <= t <= 0.5:
        print(f"Threshold: {t:.2f} => Precision: {p:.2f}, Recall: {r:.2f}")
