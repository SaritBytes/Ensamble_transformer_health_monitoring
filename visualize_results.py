import os
import joblib
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Please install matplotlib and seaborn to run this script: pip install matplotlib seaborn")
    exit(1)

from sklearn.metrics import confusion_matrix
from predict import FEATURE_COLS

DATA_PATH = r"C:\Users\abwor\Desktop\ensemble\Health_index.csv"

def generate_visualizations():
    os.makedirs('plots', exist_ok=True)
    
    print("[1] Loading Data & Models...")
    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=['Health_index'], inplace=True)
    
    def label(score):
        if score >= 75:   return 'Healthy'
        elif score >= 50: return 'About to be Unhealthy'
        else:             return 'Unhealthy'
    df['Health_Status'] = df['Health_index'].apply(label)
    
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    
    X = df[FEATURE_COLS].values
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        le = joblib.load('models/label_encoder.pkl')
        model = joblib.load('models/stack_clf.pkl')
    except Exception as e:
        print(f"Failed to load models: {e}\nPlease run 'python transformer_health_all_models.py' first.")
        return
        
    y_true = le.transform(df['Health_Status'].values)
    X_scaled = scaler.transform(X)
    
    print("[2] Generating Predictions (Full Dataset)...")
    y_pred = model.predict(X_scaled)
    
    print("[3] Plotting Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Stacking Classifier (Full Dataset)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("[4] Attempting to Plot Feature Importances...")
    # The Stacking Classifier uses LogisticRegression as a final estimator.
    # We can extract the final estimator's coefficients for feature importance across base models.
    # If we want feature importance of the raw inputs, we look into the base models (e.g., RandomForest).
    
    try:
        # Access the RandomForest base estimator
        rf_model = None
        for name, estimator in model.estimators_:
            if name == 'Random Forest':
                rf_model = estimator
                break
                
        if rf_model is not None:
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances (Random Forest Base Model)")
            plt.bar(range(X.shape[1]), importances[indices], align="center")
            plt.xticks(range(X.shape[1]), [FEATURE_COLS[i] for i in indices], rotation=45, ha='right')
            plt.xlim([-1, X.shape[1]])
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png', dpi=300)
            plt.close()
            print("    Successfully saved feature_importance.png")
        else:
            print("    Random Forest base model not found in stack_clf.")
    except Exception as e:
        print(f"    Could not plot feature importance: {e}")
        
    print(f"\n[SUCCESS] Visualizations saved to {os.path.abspath('plots')}")

if __name__ == "__main__":
    generate_visualizations()
