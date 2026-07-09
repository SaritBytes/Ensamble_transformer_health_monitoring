import sys, io
import os, joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

# ─────────────────────────────────────────────
DATA_PATH  = "Health_index.csv"

FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
]

def main():
    print("=" * 58)
    print("  Transformer Health Detection - Training Pipeline")
    print("=" * 58)
    print("\n[1] Loading dataset ...")

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    if 'Health_index' not in df.columns:
        print("Error: 'Health_index' column not found in dataset. Please ensure you have the correct CSV.")
        return

    df.dropna(subset=['Health_index'], inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"    Loaded {len(df)} records | {len(FEATURE_COLS)} features")

    def label(score):
        if score >= 75:   return 'Healthy'
        elif score >= 50: return 'About to be Unhealthy'
        else:             return 'Unhealthy'

    df['Health_Status'] = df['Health_index'].apply(label)
    
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df['Health_Status'].values)
    class_names = le.classes_

    min_count = pd.Series(y).value_counts().min()
    use_stratify = y if min_count >= 5 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=use_stratify
    )

    print(f"\n[3] Split -> Train: {len(X_train)}  |  Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n[4] Handling Data Imbalance ...")
    min_class_count = pd.Series(y_train).value_counts().min()
    k_neighbors = min(5, min_class_count - 1)
    if k_neighbors > 0:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)
    else:
        ros = RandomOverSampler(random_state=42)
        X_train_sc, y_train = ros.fit_resample(X_train_sc, y_train)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
    }

    n_samples = len(X_test)
    model_accuracies = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        acc = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = acc
        print(f"Accuracy: {acc:.4f}")

    base_estimators = [(name, model) for name, model in models.items()]
    
    print(f"\nTraining: Stacking Classifier")
    stack_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stack_clf.fit(X_train_sc, y_train)
    stack_acc = accuracy_score(y_test, stack_clf.predict(X_test_sc))
    print(f"Accuracy: {stack_acc:.4f}")

    print("\n" + "=" * 58)
    print("  SAVING MODELS TO DISK (saved_model/)")
    print("=" * 58)
    os.makedirs('saved_model', exist_ok=True)
    joblib.dump(scaler, 'saved_model/scaler.joblib')
    joblib.dump(le, 'saved_model/label_encoder.joblib')
    joblib.dump(stack_clf, 'saved_model/model.joblib')
    print("  [SUCCESS] Saved scaler, label encoder, and stack_clf to 'saved_model/' directory.")

if __name__ == "__main__":
    main()
