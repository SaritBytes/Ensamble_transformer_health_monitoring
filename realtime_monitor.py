import argparse
import time
import os
import csv
import json
import random
from collections import deque
from datetime import datetime
import pandas as pd
import joblib

FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
]

class TransformerMonitor:
    def __init__(self, source, csv_file, interval, buffer_size=1440, alert_threshold=3):
        self.source = source
        self.csv_file = csv_file
        self.interval = interval
        self.buffer = deque(maxlen=buffer_size)
        self.alert_threshold = alert_threshold
        
        self.log_file = 'prediction_log.csv'
        self._init_logger()
        
        print(f"[*] Initializing monitor (Source: {source})")
        try:
            self.scaler = joblib.load('saved_model/scaler.joblib')
            self.le = joblib.load('saved_model/label_encoder.joblib')
            self.model = joblib.load('saved_model/model.joblib')
        except Exception as e:
            print(f"Error loading models: {e}\nPlease run 'train_model.py' first to generate saved_model artifacts.")
            exit(1)
            
    def _init_logger(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp'] + FEATURE_COLS + ['predicted_status', 'confidence'])

    def log_prediction(self, record):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [record['timestamp']] + [record['features'][col] for col in FEATURE_COLS] + [record['status'], f"{record['confidence']:.2f}"]
            writer.writerow(row)

    def infer(self, features):
        df_input = pd.DataFrame([features])[FEATURE_COLS]
        X_scaled = self.scaler.transform(df_input)
        
        pred_idx = self.model.predict(X_scaled)[0]
        pred_probs = self.model.predict_proba(X_scaled)[0]
        
        status = self.le.inverse_transform([pred_idx])[0]
        confidence = pred_probs.max() * 100
        
        return status, confidence, pred_probs

    def check_alerts(self):
        if len(self.buffer) < self.alert_threshold:
            return
        
        # Check last K readings
        recent_statuses = [self.buffer[i]['status'] for i in range(-self.alert_threshold, 0)]
        if all(s == 'Unhealthy' for s in recent_statuses):
            print(f"\n[!] ALERT: Sustained 'Unhealthy' status detected over the last {self.alert_threshold} readings! [!]\n")
        elif all(s == 'About to be Unhealthy' for s in recent_statuses):
            print(f"\n[!] WARNING: Transformer is showing sustained degradation. [!]\n")

    def run(self):
        if self.source == 'simulate':
            self.run_simulation()
        else:
            print(f"Source '{self.source}' not fully implemented yet. Please use '--source simulate'.")

    def run_simulation(self):
        if not os.path.exists(self.csv_file):
            print(f"Error: CSV file '{self.csv_file}' not found.")
            return
            
        print(f"[*] Starting simulation from {self.csv_file} (interval: {self.interval}s)")
        
        # Load simulation data
        try:
            df = pd.read_csv(self.csv_file)
            # Ensure it has the right columns (for this demonstration, if it doesn't we'll just skip to avoid crashing immediately, but it should fail gracefully during iter)
        except Exception as e:
            print(f"Failed to read CSV: {e}")
            return
            
        for _, row in df.iterrows():
            features = {col: row.get(col, 0.0) for col in FEATURE_COLS}
            
            timestamp = datetime.now().isoformat()
            
            try:
                status, confidence, probs = self.infer(features)
            except Exception as e:
                print(f"Inference failed (likely missing columns in dataset): {e}")
                time.sleep(self.interval)
                continue
            
            record = {
                'timestamp': timestamp,
                'features': features,
                'status': status,
                'confidence': confidence,
                'probabilities': {cls: float(p) for cls, p in zip(self.le.classes_, probs)}
            }
            
            self.buffer.append(record)
            self.log_prediction(record)
            
            print(f"[{timestamp}] Pred: {status:<22} | Conf: {confidence:>6.2f}% | Buffer: {len(self.buffer)}/{self.buffer.maxlen}")
            
            self.check_alerts()
            time.sleep(self.interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Transformer Health Monitor")
    parser.add_argument('--source', type=str, choices=['modbus', 'mqtt', 'simulate'], default='simulate', help='Data source')
    parser.add_argument('--csv-file', type=str, default='Health index.csv', help='Path to historical CSV for simulation')
    parser.add_argument('--interval', type=int, default=2, help='Polling interval in seconds')
    
    args = parser.parse_args()
    
    monitor = TransformerMonitor(source=args.source, csv_file=args.csv_file, interval=args.interval)
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n[*] Monitor stopped.")
