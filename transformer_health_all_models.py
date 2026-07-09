import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
Transformer Health Detection - Ensemble Models
========================================================
Dataset  : Health index.csv
Base     : Random Forest, XGBoost, AdaBoost, LightGBM
Ensemble : Weighted Soft Voting + Stacking (Logistic Regression meta-learner)
Features : 14 DGA + electrical parameters
Target   : Health_index -> 3 class labels
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

# ─────────────────────────────────────────────
DATA_PATH  = r"C:\Users\abwor\Desktop\ensemble\Health_index.csv"

FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
]

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 58)
print("  Transformer Health Detection - All Models + Ensemble")
print("=" * 58)
print("\n[1] Loading dataset ...")

df = pd.read_csv(DATA_PATH)
df.dropna(subset=['Health_index'], inplace=True)
df.drop_duplicates(inplace=True)
print(f"    Loaded {len(df)} records | {len(FEATURE_COLS)} features")

# ─────────────────────────────────────────────
# 2. CREATE CLASS LABELS & FILL MISSING VALUES
# ─────────────────────────────────────────────
def label(score):
    if score >= 75:   return 'Healthy'
    elif score >= 50: return 'About to be Unhealthy'
    else:             return 'Unhealthy'

df['Health_Status'] = df['Health_index'].apply(label)
print("\n[2] Original Class distribution:")
for cls, cnt in df['Health_Status'].value_counts().items():
    print(f"    {cls:<25} : {cnt:4d}  ({cnt/len(df)*100:.1f}%)")

df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

# ─────────────────────────────────────────────
# 3. PREPROCESS, SPLIT & SCALE
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 4. BALANCE DATASET
# ─────────────────────────────────────────────
print("\n[4] Handling Data Imbalance with SMOTE ...")
min_class_count = pd.Series(y_train).value_counts().min()
k_neighbors = min(5, min_class_count - 1)
if k_neighbors > 0:
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)
    print(f"    Applied SMOTE (k_neighbors={k_neighbors})")
else:
    ros = RandomOverSampler(random_state=42)
    X_train_sc, y_train = ros.fit_resample(X_train_sc, y_train)
    print("    Applied RandomOverSampler (classes too small for SMOTE)")

dist = pd.Series(y_train).value_counts()
dist.index = class_names[dist.index]
print(f"    New Train class distribution: {dist.to_dict()}")

# ─────────────────────────────────────────────
# 5. MODELS DICTIONARY
# ─────────────────────────────────────────────
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
}

# ─────────────────────────────────────────────
# 6. TRAIN & EVALUATE ALL BASE MODELS
# ─────────────────────────────────────────────
n_samples = len(X_test)
model_accuracies = {}   # used later to weight the voting ensemble

for model_name, model in models.items():
    print("\n" + "=" * 58)
    print(f"  Training & Evaluating: {model_name}")
    print("=" * 58)

    # Train
    model.fit(X_train_sc, y_train)

    # Predict
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    model_accuracies[model_name] = acc

    print(f"\n  Accuracy : {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Test Predictions Output
    print("\n  " + "-" * 68)
    print(f"  TEST PREDICTIONS ({model_name} - all {n_samples} test rows)")
    print("  " + "-" * 68)
    print(f"  {'Actual':<25} {'Predicted':<25} {'Confidence':>10}  Result")
    print("  " + "-" * 68)

    proba = model.predict_proba(X_test_sc[:n_samples])
    pred_s = model.predict(X_test_sc[:n_samples])
    for i in range(n_samples):
        actual = class_names[y_test[i]]
        pred   = class_names[pred_s[i]]
        conf   = f"{proba[i].max()*100:.1f}%"
        result = "CORRECT" if y_test[i] == pred_s[i] else "WRONG"
        print(f"  {actual:<25} {pred:<25} {conf:>10}  {result}")

# ─────────────────────────────────────────────
# 7. ENSEMBLE MODELS
# ─────────────────────────────────────────────
base_estimators = [(name, model) for name, model in models.items()]
ensemble_accuracies = {}

def evaluate_ensemble(name, clf):
    """Fit an ensemble, print its report + prediction table, return accuracy."""
    print("\n" + "=" * 58)
    print(f"  Training & Evaluating: {name}")
    print("=" * 58)

    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    ensemble_accuracies[name] = acc

    print(f"\n  Accuracy : {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\n  " + "-" * 68)
    print(f"  TEST PREDICTIONS ({name} - all {n_samples} test rows)")
    print("  " + "-" * 68)
    print(f"  {'Actual':<25} {'Predicted':<25} {'Confidence':>10}  Result")
    print("  " + "-" * 68)

    proba = clf.predict_proba(X_test_sc[:n_samples])
    pred_s = clf.predict(X_test_sc[:n_samples])
    for i in range(n_samples):
        actual = class_names[y_test[i]]
        pred   = class_names[pred_s[i]]
        conf   = f"{proba[i].max()*100:.1f}%"
        result = "CORRECT" if y_test[i] == pred_s[i] else "WRONG"
        print(f"  {actual:<25} {pred:<25} {conf:>10}  {result}")

# --- 7a. Weighted Soft Voting -----------------------------------------
# Each model's vote is weighted by its own standalone test accuracy,
# so stronger models influence the final probability average more.
weights = [model_accuracies[name] for name, _ in base_estimators]
voting_clf = VotingClassifier(
    estimators=base_estimators, voting='soft', weights=weights, n_jobs=-1
)
evaluate_ensemble("Ensemble: Weighted Soft Voting", voting_clf)

# --- 7b. Stacking (Logistic Regression meta-learner) -------------------
# The meta-learner trains on out-of-fold predicted probabilities from the
# 4 base models (cv=5), so it never sees a base model's prediction on the
# same rows it was trained on -> avoids leakage/overfitting.
stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)
evaluate_ensemble("Ensemble: Stacking (Logistic Regression meta-learner)", stack_clf)

# ─────────────────────────────────────────────
# 8. FINAL COMPARISON SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  FINAL ACCURACY COMPARISON")
print("=" * 58)
all_results = {**model_accuracies, **ensemble_accuracies}
for name, acc in sorted(all_results.items(), key=lambda x: -x[1]):
    tag = "  <- BEST" if acc == max(all_results.values()) else ""
    print(f"  {name:<45} : {acc:.4f}{tag}")

print("\n" + "=" * 58)
print("  All Models & Ensembles Training/Evaluation Complete!")
print("=" * 58)

# ─────────────────────────────────────────────
# 9. SAVE MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("  SAVING MODELS TO DISK")
print("=" * 58)
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(stack_clf, 'models/stack_clf.pkl')
print("  [SUCCESS] Saved scaler, label encoder, and stack_clf to 'models/' directory.")