# Transformer Health Detection - Ensemble Models

This project provides a robust machine learning pipeline for diagnosing the health status of electrical power transformers using Dissolved Gas Analysis (DGA) and electrical parameter datasets. To combat data imbalance and prevent model bias, it implements data augmentation (SMOTE) and evaluates a suite of advanced tree-based classifiers alongside two sophisticated ensemble meta-models.

## Features

- **Data Imputation & Preprocessing:** Handles missing values and scales features automatically.
- **Class Balancing:** Utilizes Synthetic Minority Over-sampling Technique (SMOTE) and RandomOverSampler to balance underrepresented failure scenarios.
- **Base Classifiers:** Includes Random Forest, XGBoost, AdaBoost, and LightGBM models.
- **Ensemble Architectures:** 
  - **Weighted Soft Voting:** Aggregates class probabilities based on individual model test accuracies.
  - **Stacking Classifier:** Uses a Logistic Regression meta-learner trained on 5-fold cross-validated out-of-fold predictions to deduce the final diagnosis.
- **Performance Evaluation:** Outputs detailed classification reports and side-by-side confidence metrics for every test sample.

## Requirements

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
```

## Dataset

The script expects a dataset named `Health_index.csv` in the same working directory. 
The dataset must contain the continuous target column `Health_index` and the following 14 features:
- **Chemical/Gases:** Hydrogen, Oxigen, Nitrogen, Methane, CO, CO2, Ethylene, Ethane, Acethylene, DBDS
- **Electrical/Physical:** Power_factor, Interfacial_V, Dielectric_rigidity, Water_content

The `Health_index` target variable is automatically discretized into three categories:
- **Healthy:** $\ge 75$
- **About to be Unhealthy:** $50 - 74$
- **Unhealthy:** $< 50$

## Usage

Run the main pipeline directly from your terminal or command prompt:

```bash
python transformer_health_all_models.py
```

### Execution Flow:
1. **Load Data:** Imports `Health_index.csv` and removes missing target labels.
2. **Preprocess:** Categorizes the health index, imputes missing feature values, and scales using `StandardScaler`.
3. **Balance:** Balances training data via SMOTE to equalize class representations.
4. **Base Training:** Trains RF, XGB, AdaBoost, and LightGBM, and prints their standalone performance metrics.
5. **Ensemble Training:** Combines models using Weighted Soft Voting and Stacking methodologies.
6. **Final Comparison:** Outputs a sorted leaderboard of all models based on their test accuracy.

## Project Structure

```
ensemble/
│
├── transformer_health_all_models.py   # Main pipeline script
├── Health_index.csv                   # Dataset (Provide this file)
└── README.md                          # This documentation file
```

## License

This project is open-source and available for educational and research purposes.
