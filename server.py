"""
FastAPI Backend Server - Transformer Health Monitor
=====================================================
Hosts the in-memory FIFO ring buffer, inference engine,
alert engine, and serves the static dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn

# ── Constants (will be overwritten from saved artifacts if available) ──
DEFAULT_FEATURES = ['Water', 'Acidity', 'DBV', 'DF', 'TDCG', 'Furan']

# ── Global State ──
BUFFER_SIZE = 1440
memory_buffer: deque = deque(maxlen=BUFFER_SIZE)
alert_state = {"active": False, "message": "", "level": "ok"}
ALERT_THRESHOLD = 3

# ── Load ML Models ──
models_loaded = False
scaler = None
le = None
model = None
FEATURE_COLS = DEFAULT_FEATURES

try:
    scaler = joblib.load('saved_model/scaler.joblib')
    le = joblib.load('saved_model/label_encoder.joblib')
    model = joblib.load('saved_model/model.joblib')
    if os.path.exists('saved_model/feature_cols.joblib'):
        FEATURE_COLS = joblib.load('saved_model/feature_cols.joblib')
    models_loaded = True
    print(f"[OK] Models loaded. Features: {FEATURE_COLS}")
    print(f"[OK] Classes: {list(le.classes_)}")
except Exception as e:
    print(f"[WARN] Models not loaded: {e}")
    print("       Run 'python train_model.py' first.")

# ── Pydantic Request Model ──
class SensorPayload(BaseModel):
    features: Dict[str, float]

# ── FastAPI App ──
app = FastAPI(title="Transformer Health Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/api/features")
def get_features():
    """Returns expected feature column names so the frontend knows what to display."""
    return {"features": FEATURE_COLS, "models_loaded": models_loaded}

@app.post("/api/ingest")
async def ingest_sensor_data(payload: SensorPayload):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")

    features = payload.features
    for col in FEATURE_COLS:
        if col not in features:
            raise HTTPException(status_code=400, detail=f"Missing feature: {col}")

    # Inference
    df_input = pd.DataFrame([features])[FEATURE_COLS]
    X_scaled = scaler.transform(df_input)

    pred_idx = model.predict(X_scaled)[0]
    pred_probs = model.predict_proba(X_scaled)[0]

    status = le.inverse_transform([pred_idx])[0]
    confidence = float(pred_probs.max() * 100)

    # Map short labels to readable names
    label_map = {'VG': 'Very Good', 'G': 'Good', 'M': 'Moderate', 'B': 'Bad', 'VB': 'Very Bad'}
    readable_status = label_map.get(status, status)

    timestamp = datetime.now().isoformat()
    record = {
        'timestamp': timestamp,
        'features': features,
        'status': status,
        'readable_status': readable_status,
        'confidence': round(confidence, 2),
        'probabilities': {cls: round(float(p) * 100, 2) for cls, p in zip(le.classes_, pred_probs)}
    }

    memory_buffer.append(record)

    # ── Alert Engine ──
    if len(memory_buffer) >= ALERT_THRESHOLD:
        recent = [memory_buffer[i]['status'] for i in range(-ALERT_THRESHOLD, 0)]
        if all(s in ('B', 'VB') for s in recent):
            alert_state["active"] = True
            alert_state["message"] = f"CRITICAL: Last {ALERT_THRESHOLD} readings show Bad/Very Bad health!"
            alert_state["level"] = "danger"
        elif all(s == 'M' for s in recent):
            alert_state["active"] = True
            alert_state["message"] = f"WARNING: Sustained Moderate health over {ALERT_THRESHOLD} readings."
            alert_state["level"] = "warning"
        else:
            alert_state["active"] = False
            alert_state["message"] = ""
            alert_state["level"] = "ok"

    return {"status": "ok", "prediction": record}

@app.get("/api/history")
async def get_history():
    return list(memory_buffer)

@app.get("/api/alerts")
async def get_alerts():
    return alert_state

@app.get("/api/stats")
async def get_stats():
    """Summary statistics from the FIFO buffer."""
    if not memory_buffer:
        return {"buffer_size": 0, "buffer_capacity": BUFFER_SIZE}

    statuses = [r['status'] for r in memory_buffer]
    return {
        "buffer_size": len(memory_buffer),
        "buffer_capacity": BUFFER_SIZE,
        "status_counts": {s: statuses.count(s) for s in set(statuses)},
        "avg_confidence": round(np.mean([r['confidence'] for r in memory_buffer]), 2),
        "latest_timestamp": memory_buffer[-1]['timestamp'],
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
