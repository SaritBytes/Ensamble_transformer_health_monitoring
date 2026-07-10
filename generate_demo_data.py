"""
Generate Demo Data for Static Deployment
==========================================
Loads the trained model and runs inference on all rows of Health_index.csv.
Outputs static/demo_data.json with predictions, probabilities, and features.
"""

import json
import os
import pandas as pd
import numpy as np
import joblib

FEATURE_COLS = ['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan']
LABEL_MAP = {'VG': 'Very Good', 'G': 'Good', 'M': 'Moderate', 'B': 'Bad', 'VB': 'Very Bad'}

def main():
    print("=" * 58)
    print("  Generating Demo Data for Static Dashboard")
    print("=" * 58)

    # Load models
    scaler = joblib.load('saved_model/scaler.joblib')
    le = joblib.load('saved_model/label_encoder.joblib')
    model = joblib.load('saved_model/model.joblib')

    if os.path.exists('saved_model/feature_cols.joblib'):
        feature_cols = joblib.load('saved_model/feature_cols.joblib')
    else:
        feature_cols = FEATURE_COLS

    print(f"  Features: {feature_cols}")
    print(f"  Classes:  {list(le.classes_)}")

    # Load CSV
    df = pd.read_csv('Health_index.csv')
    print(f"  CSV rows: {len(df)}")

    records = []
    for idx, row in df.iterrows():
        features = {}
        for col in feature_cols:
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            features[col] = float(val)

        # Inference
        df_input = pd.DataFrame([features])[feature_cols]
        X_scaled = scaler.transform(df_input)
        pred_idx = model.predict(X_scaled)[0]
        pred_probs = model.predict_proba(X_scaled)[0]

        status = le.inverse_transform([pred_idx])[0]
        confidence = float(pred_probs.max() * 100)
        readable_status = LABEL_MAP.get(status, status)

        record = {
            'features': features,
            'status': status,
            'readable_status': readable_status,
            'confidence': round(confidence, 2),
            'probabilities': {
                cls: round(float(p) * 100, 2)
                for cls, p in zip(le.classes_, pred_probs)
            }
        }
        records.append(record)
        print(f"  [{idx+1:3d}] {readable_status:<12} ({confidence:.1f}%)")

    # Write JSON
    output_path = os.path.join('static', 'demo_data.json')
    with open(output_path, 'w') as f:
        json.dump({
            'features': feature_cols,
            'classes': list(le.classes_),
            'records': records
        }, f, indent=2)

    print(f"\n  [OK] Wrote {len(records)} records to {output_path}")

if __name__ == '__main__':
    main()
