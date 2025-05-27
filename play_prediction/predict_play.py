import pandas as pd
import os 
from pathlib import Path
from joblib import load

base_dir = Path(__file__).resolve().parent
model_path = base_dir / "xgb_model.joblib"
scaler_path = base_dir / "scaler.joblib"

model = load(model_path)
scaler = load(scaler_path)

feature_columns = [
    "down",
    "ydstogo",
    "half_seconds_remaining",
    "score_differential",
    "yardline_100",
    "pass_pct_last_20",
    "pass_pct_diff_10_vs_40"
]
scale_cols = [
    "down",
    "ydstogo",
    "half_seconds_remaining",
    "score_differential",
    "yardline_100"
]

def predict_single_play(play_row: pd.Series) -> str:
    if play_row[feature_columns].isnull().any():
        return f"Play ID {play_row['id']}: Missing data, unable to predict."

    play_df = pd.DataFrame([play_row])

    play_df_scaled = play_df.copy()
    play_df_scaled[scale_cols] = scaler.transform(play_df[scale_cols])

    X = play_df_scaled[feature_columns].values

    y_pred_proba = model.predict_proba(X)[0]
    y_pred = model.predict(X)[0]
    
    label = "Pass" if y_pred == 1 else "Run"
    confidence = round(y_pred_proba[y_pred], 4)
    return f"PREDICTION: {label}, CONFIDENCE: {confidence:.3f}"

