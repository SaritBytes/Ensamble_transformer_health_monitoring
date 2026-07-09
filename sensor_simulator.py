import argparse
import time
import requests
import pandas as pd
import json

FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
]

def run_simulator(csv_file, endpoint, interval):
    print(f"[*] Starting Data Simulator")
    print(f"    Reading from: {csv_file}")
    print(f"    Sending to: {endpoint}")
    print(f"    Interval: {interval}s\n")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Keep looping over the data forever to simulate a continuous stream
    while True:
        for _, row in df.iterrows():
            # Build payload
            # If the CSV has completely different columns (like the user's current fuzzy logic dataset), 
            # this will send NaNs or default to 0.0, but the API expects these exact columns.
            features = {}
            for col in FEATURE_COLS:
                val = row.get(col, 0.0)
                # Ensure it's a valid float (NaN -> 0.0)
                if pd.isna(val): val = 0.0
                features[col] = float(val)
                
            payload = {"features": features}
            
            try:
                res = requests.post(endpoint, json=payload, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    status = data.get('prediction', {}).get('status', 'Unknown')
                    print(f"[SUCCESS] Sent record. Server predicted: {status}")
                else:
                    print(f"[ERROR] API returned status {res.status_code}: {res.text}")
            except requests.exceptions.ConnectionError:
                print(f"[ERROR] Failed to connect to {endpoint}. Is the server running?")
            except Exception as e:
                print(f"[ERROR] Request failed: {e}")
                
            time.sleep(interval)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='Health index.csv')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:8000/api/ingest')
    parser.add_argument('--interval', type=float, default=2.0)
    
    args = parser.parse_args()
    try:
        run_simulator(args.csv, args.endpoint, args.interval)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
