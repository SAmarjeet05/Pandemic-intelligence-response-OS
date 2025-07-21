# predict.py

import pandas as pd
from pathlib import Path
import joblib
import argparse
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Argument parser for optional evaluation flag
parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance on test data')
args = parser.parse_args()

base_path = Path(__file__).resolve().parent.parent 
data_path = base_path / "data" / "raw"
model_path = base_path / "model"
processed_data_path = base_path / "data" / "processed"

# Load model
model = joblib.load(model_path / "auto_trained_model.pkl")

# Load data
df = pd.read_csv(processed_data_path / "final_dataset_with_labels.csv")

# Feature columns
feature_cols = [
    'cluster_score', 'mobility_index', 'residential_change', 'transport_usage',
    'icu_usage_per_1000', 'tweet_volume', 'sentiment_score', 'temp_avg',
    'humidity', 'air_quality_index', 'rainfall_mm',
    'keyword_0', 'keyword_1', 'keyword_2', 'keyword_3', 'keyword_4'
]

# Prepare input features
X = df[feature_cols]
y_true = df['outbreak'].astype(int)

# Make predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]  # Probability of class 1

# Create a results DataFrame
results_df = df[['date', 'district_id']].copy()
results_df['Predicted Outbreak'] = y_pred
results_df['Probability (%)'] = (y_prob * 100).round(2)
results_df['Actual Outbreak'] = y_true

# Display a few sample predictions
print("📊 Sample Outbreak Predictions:\n")
print(results_df.head(10).to_string(index=False))

# Save predictions
results_df.to_csv(processed_data_path / "predictions.csv", index=False)
print("\n✅ Predictions saved to data/processed/predictions.csv")

# Optional: Evaluate if --evaluate flag passed
if args.evaluate:
    print("\n📈 Model Evaluation:\n")
    print(classification_report(y_true, y_pred))
    print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.2f}")

    # Heatmap of confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
