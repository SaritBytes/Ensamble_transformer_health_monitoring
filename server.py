from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
import pandas as pd
import joblib
import os
import uvicorn

# --- Pydantic Models for Validation ---
class SensorPayload(BaseModel):
    features: Dict[str, float]
    
# --- Constants ---
FEATURE_COLS = [
    'Hydrogen', 'Oxigen', 'Nitrogen', 'Methane', 'CO', 'CO2',
    'Ethylene', 'Ethane', 'Acethylene', 'DBDS',
    'Power_factor', 'Interfacial_V', 'Dielectric_rigidity', 'Water_content'
]

# --- Global State (FIFO Ring Buffer) ---
BUFFER_SIZE = 1440 # 24 hours of data at 1 reading per minute
memory_buffer = deque(maxlen=BUFFER_SIZE)
alert_state = {"active": False, "message": ""}

# --- Load ML Models ---
try:
    scaler = joblib.load('saved_model/scaler.joblib')
    le = joblib.load('saved_model/label_encoder.joblib')
    model = joblib.load('saved_model/model.joblib')
    models_loaded = True
except Exception as e:
    print(f"Warning: Failed to load models. {e}")
    models_loaded = False

# --- FastAPI App ---
app = FastAPI(title="Transformer Health API")

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/api/ingest")
async def ingest_sensor_data(payload: SensorPayload):
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Validate features
    features = payload.features
    for col in FEATURE_COLS:
        if col not in features:
            raise HTTPException(status_code=400, detail=f"Missing feature: {col}")
            
    # Inference
    try:
        df_input = pd.DataFrame([features])[FEATURE_COLS]
        X_scaled = scaler.transform(df_input)
        
        pred_idx = model.predict(X_scaled)[0]
        pred_probs = model.predict_proba(X_scaled)[0]
        
        status = le.inverse_transform([pred_idx])[0]
        confidence = float(pred_probs.max() * 100)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Create Record
    timestamp = datetime.now().isoformat()
    record = {
        'timestamp': timestamp,
        'features': features,
        'status': status,
        'confidence': confidence,
        'probabilities': {cls: float(p) for cls, p in zip(le.classes_, pred_probs)}
    }
    
    # Store in FIFO buffer
    memory_buffer.append(record)
    
    # Alert Engine logic (check last 3 readings)
    alert_threshold = 3
    if len(memory_buffer) >= alert_threshold:
        recent_statuses = [memory_buffer[i]['status'] for i in range(-alert_threshold, 0)]
        if all(s == 'Unhealthy' for s in recent_statuses):
            alert_state["active"] = True
            alert_state["message"] = "Sustained critical failure detected!"
        elif all(s == 'About to be Unhealthy' for s in recent_statuses):
            alert_state["active"] = True
            alert_state["message"] = "Sustained degradation detected."
        else:
            alert_state["active"] = False
            alert_state["message"] = ""
            
    return {"status": "success", "prediction": record}

@app.get("/api/history")
async def get_history():
    return list(memory_buffer)

@app.get("/api/alerts")
async def get_alerts():
    return alert_state

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
