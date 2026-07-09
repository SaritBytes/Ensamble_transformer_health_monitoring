"""
Sensor Simulator - Feeds historical CSV data into the API
==========================================================
Reads rows from Health_index.csv and POSTs them as JSON
payloads to the FastAPI /api/ingest endpoint, mimicking
a physical IoT sensor stream.
"""

import argparse
import time
import requests
import pandas as pd

FEATURE_COLS = ['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan']

def run_simulator(csv_file, endpoint, interval):
    print("=" * 58)
    print("  Sensor Data Simulator")
    print("=" * 58)
    print(f"  CSV      : {csv_file}")
    print(f"  Endpoint : {endpoint}")
    print(f"  Interval : {interval}s")
    print(f"  Features : {FEATURE_COLS}")
    print()

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Continuous loop (replays CSV endlessly)
    cycle = 1
    while True:
        print(f"--- Cycle {cycle} ({len(df)} records) ---")
        for idx, row in df.iterrows():
            features = {}
            for col in FEATURE_COLS:
                val = row.get(col, 0.0)
                if pd.isna(val):
                    val = 0.0
                features[col] = float(val)

            payload = {"features": features}

            try:
                res = requests.post(endpoint, json=payload, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    pred = data.get('prediction', {})
                    status = pred.get('readable_status', 'Unknown')
                    conf = pred.get('confidence', 0)
                    print(f"  [{idx+1:3d}/{len(df)}] -> {status:<12} ({conf:.1f}%)")
                else:
                    print(f"  [{idx+1:3d}/{len(df)}] ERROR {res.status_code}: {res.text[:80]}")
            except requests.exceptions.ConnectionError:
                print(f"  Connection failed! Is the server running at {endpoint}?")
            except Exception as e:
                print(f"  Request error: {e}")

            time.sleep(interval)

        cycle += 1
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sensor Data Simulator")
    parser.add_argument('--csv', type=str, default='Health_index.csv')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:8000/api/ingest')
    parser.add_argument('--interval', type=float, default=2.0)

    args = parser.parse_args()
    try:
        run_simulator(args.csv, args.endpoint, args.interval)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
