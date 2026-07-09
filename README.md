# Real-Time Transformer Health Monitoring System

This project provides a robust machine learning pipeline and a real-time monitoring system for diagnosing the health status of electrical power transformers using Dissolved Gas Analysis (DGA) and electrical parameter datasets. 

To combat data imbalance and prevent model bias, it implements data augmentation (SMOTE) and evaluates a suite of advanced tree-based classifiers alongside a Stacking Classifier meta-model.

## System Architecture

The project is split into two distinct phases:

### Phase 1: Training (`train_model.py`)
This script handles the heavy lifting of reading historical data, preprocessing, scaling, balancing via SMOTE, and training multiple classifiers. The best-performing ensemble (Stacking Classifier) is then exported to the `saved_model/` directory alongside its data scaler and label encoder.

### Phase 2: Real-Time Monitoring (`realtime_monitor.py`)
This script acts as a live, long-running daemon. It uses an in-memory **FIFO Ring Buffer** (`collections.deque`) to store incoming sensor readings. It runs real-time inference on the newest readings using the pre-trained models from Phase 1. 

**Key Features of Phase 2:**
- **In-Memory Operations:** Prevents slow disk I/O during critical inference.
- **Alert Engine:** Triggers alerts only if sustained faults are detected (e.g., 3 consecutive "Unhealthy" readings), minimizing false alarms.
- **Disk Archiving:** Quietly appends all predictions to a background `prediction_log.csv` file for compliance and long-term storage.
- **Simulation Mode:** Allows you to test the pipeline by reading a historical CSV at a defined polling interval, simulating real-world sensor streams.

## Requirements

Ensure you have Python 3.8+ installed. You can install all required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset Requirements

The system expects a dataset named `Health index.csv` containing the continuous target column `Health_index` and the following 14 features:
- **Chemical/Gases:** Hydrogen, Oxigen, Nitrogen, Methane, CO, CO2, Ethylene, Ethane, Acethylene, DBDS
- **Electrical/Physical:** Power_factor, Interfacial_V, Dielectric_rigidity, Water_content

The `Health_index` target variable is automatically discretized into three categories:
- **Healthy:** $\ge 75$
- **About to be Unhealthy:** $50 - 74$
- **Unhealthy:** $< 50$

> **Note:** If your CSV has different column names, the scripts will crash. Ensure the dataset matches the 14 DGA parameters exactly.

## Usage Guide

### 1. Train the Model
Run the training script once (or whenever you get new historical data) to generate the model artifacts:

```bash
python train_model.py
```
*Expected output: `scaler.joblib`, `label_encoder.joblib`, and `model.joblib` will be generated inside the `saved_model/` folder.*

### 2. Run the Real-Time Monitor
Start the monitoring daemon. To simulate real-time sensor data using your historical CSV, use the `--source simulate` flag:

```bash
python realtime_monitor.py --source simulate --csv-file "Health index.csv" --interval 2
```
*This will ingest one row from the CSV every 2 seconds, run inference, update the FIFO buffer, check for sustained alerts, and log the results.*

### Optional: Legacy Scripts
- `transformer_health_all_models.py`: The original monolithic script that trains and evaluates all models in one go.
- `predict.py`: A simple CLI inference script to test single, one-off dictionary predictions.
- `visualize_results.py`: Generates confusion matrices and feature importance graphs for your project report.

## Project Structure

```
ensemble/
│
├── train_model.py                 # Phase 1: Training script
├── realtime_monitor.py            # Phase 2: Live FIFO buffer monitoring script
├── requirements.txt               # Python dependencies
├── Health index.csv               # Historical dataset (Ensure columns match)
├── README.md                      # This documentation file
│
├── saved_model/                   # Generated during training
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── model.joblib
│
└── legacy/ (Optional)
    ├── transformer_health_all_models.py
    ├── predict.py
    └── visualize_results.py
```
