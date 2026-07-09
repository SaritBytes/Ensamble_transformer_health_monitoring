"""
Transformer Health Detection - Training Pipeline
=================================================
Dataset  : Health_index.csv
Features : Water, Acidity, DBV, DF, TDCG, Furan (6 oil-quality parameters)
Target   : Fuzzy health label (VG=Very Good, G=Good, M=Moderate, B=Bad, VB=Very Bad)
Base     : Random Forest, XGBoost, AdaBoost, LightGBM
Ensemble : Stacking (Logistic Regression meta-learner)
Output   : saved_model/ (scaler.joblib, label_encoder.joblib, model.joblib)
"""

import os, joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

# ─────────────────────────────────────────────
DATA_PATH = "Health_index.csv"

FEATURE_COLS = ['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan']
TARGET_COL = 'Fuzzy'  # VG, G, M, B, VB

def main():
    print("=" * 58)
    print("  Transformer Health Detection - Training Pipeline")
    print("=" * 58)

    # 1. Load
    print("\n[1] Loading dataset ...")
    if not os.path.exists(DATA_PATH):
        print(f"    Error: {DATA_PATH} not found."); return

    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=[TARGET_COL], inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"    Loaded {len(df)} records | {len(FEATURE_COLS)} features | target='{TARGET_COL}'")

    # 2. Preprocess
    print("\n[2] Class distribution:")
    for cls, cnt in df[TARGET_COL].value_counts().items():
        print(f"    {cls:<15} : {cnt:4d}  ({cnt/len(df)*100:.1f}%)")

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL].values)
    class_names = le.classes_
    print(f"    Encoded classes: {list(class_names)}")

    # 3. Split
    min_count = pd.Series(y).value_counts().min()
    use_stratify = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=use_stratify
    )
    print(f"\n[3] Split -> Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 5. Balance
    print("\n[4] Handling Data Imbalance ...")
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
    print(f"    Balanced distribution: {dist.to_dict()}")

    # 6. Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, random_state=42,
                                  n_jobs=-1, eval_metric='mlogloss'),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
    }

    # 7. Train & evaluate base models
    model_accuracies = {}
    for name, mdl in models.items():
        mdl.fit(X_train_sc, y_train)
        acc = accuracy_score(y_test, mdl.predict(X_test_sc))
        model_accuracies[name] = acc
        print(f"\n  {name:<20} Accuracy: {acc:.4f}")
        print(classification_report(y_test, mdl.predict(X_test_sc),
                                     target_names=class_names, zero_division=0))

    # 8. Stacking ensemble
    print("\n[5] Training Stacking Classifier ...")
    base_estimators = list(models.items())
    stack_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=min(5, min_class_count) if min_class_count >= 2 else 2,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stack_clf.fit(X_train_sc, y_train)
    stack_acc = accuracy_score(y_test, stack_clf.predict(X_test_sc))
    print(f"  Stacking Accuracy: {stack_acc:.4f}")
    print(classification_report(y_test, stack_clf.predict(X_test_sc),
                                 target_names=class_names, zero_division=0))

    # 9. Save
    print("=" * 58)
    print("  SAVING MODELS")
    print("=" * 58)
    os.makedirs('saved_model', exist_ok=True)
    joblib.dump(scaler, 'saved_model/scaler.joblib')
    joblib.dump(le, 'saved_model/label_encoder.joblib')
    joblib.dump(stack_clf, 'saved_model/model.joblib')
    # Also save feature column list for the server
    joblib.dump(FEATURE_COLS, 'saved_model/feature_cols.joblib')
    print("  [OK] Saved to saved_model/")

if __name__ == "__main__":
    main()
