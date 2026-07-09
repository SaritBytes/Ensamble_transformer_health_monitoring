import os
import joblib
import pandas as pd
import numpy as np

# Feature names exactly as used during training
FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
]

def load_models(models_dir='models'):
    """Loads the scaler, label encoder, and stacking classifier from disk."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found. Please run the training script first.")
    
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    model = joblib.load(os.path.join(models_dir, 'stack_clf.pkl'))
    
    return scaler, le, model

def predict_health(input_data, scaler, le, model):
    """
    Predicts the transformer health status given a dictionary of input parameters.
    """
    # Ensure input_data contains all required features
    for col in FEATURE_COLS:
        if col not in input_data:
            raise ValueError(f"Missing required feature: {col}")
            
    # Convert to DataFrame to ensure correct order
    df_input = pd.DataFrame([input_data])[FEATURE_COLS]
    
    # Scale the input
    X_scaled = scaler.transform(df_input)
    
    # Predict
    pred_idx = model.predict(X_scaled)[0]
    pred_probs = model.predict_proba(X_scaled)[0]
    
    # Decode label
    health_status = le.inverse_transform([pred_idx])[0]
    confidence = pred_probs.max() * 100
    
    return health_status, confidence, pred_probs

if __name__ == '__main__':
    print("=" * 58)
    print("  Transformer Health Prediction (Inference)")
    print("=" * 58)
    
    try:
        scaler, le, model = load_models()
        
        # Sample realistic data for a healthy transformer
        sample_data = {
            'Hydrogen': 15.0,
            'Oxigen': 300.0,
            'Nitrogen': 45000.0,
            'Methane': 5.0,
            'CO': 150.0,
            'CO2': 1200.0,
            'Ethylene': 2.0,
            'Ethane': 10.0,
            'Acethylene': 0.0,
            'DBDS': 0.0,
            'Power_factor': 0.002,
            'Interfacial_V': 45.0,
            'Dielectric_rigidity': 65.0,
            'Water_content': 12.0
        }
        
        print("\nInput Parameters:")
        for k, v in sample_data.items():
            print(f"  {k:<20}: {v}")
            
        status, conf, probs = predict_health(sample_data, scaler, le, model)
        
        print("\n" + "-" * 58)
        print(f"  PREDICTION : {status}")
        print(f"  CONFIDENCE : {conf:.2f}%")
        print("-" * 58)
        
        print("\nProbabilities by class:")
        for class_name, prob in zip(le.classes_, probs):
            print(f"  {class_name:<25}: {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Error: {e}")
