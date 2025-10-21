
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Optional
from .telemetry import Telemetry
from .engine_v5 import CognitiveEngineV5

app = FastAPI(title="Suraj Self-Evolving Quant AI v5", version="1.0.0")
telemetry = Telemetry()
engine = CognitiveEngineV5({
    "sequence_length": 48,
    "input_size": 12,
    "lstm_hidden": 48,
    "tr_d_model": 64,
    "registry_root": "models/v5"
})
engine.load_best()

class PredictIn(BaseModel):
    symbol: str = Field(..., example="XAUUSD")
    price: float = Field(..., example=2350.5)
    equity: float = Field(..., example=2000.0)
    regime: Optional[str] = "neutral"
    drift: Optional[float] = 0.0
    features: Dict = Field(default_factory=dict)

@app.get("/status")
def status():
    return {"ok": True, "engine": "v5", "registry": engine.registry_info()}

@app.post("/predict")
def predict(inp: PredictIn):
    f = dict(inp.features)
    f["sentiment"] = f.get("sentiment", 0.0)
    f["macro_state"] = f.get("macro_state", 0.0)
    f["drift"] = float(inp.drift or 0.0)
    f["regime"] = inp.regime
    out = engine.predict(f, inp.price)
    telemetry.event("predict", symbol=inp.symbol, price=inp.price, out=out)
    return out

@app.post("/train_online")
def train_online(steps: int = 1):
    res = engine.online_train(steps=steps)
    telemetry.event("online_train", **res)
    return res

@app.post("/retrain")
def retrain():
    res = engine.retrain_and_evaluate()
    telemetry.event("retrain", **res)
    return res
