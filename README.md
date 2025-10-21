
# Suraj Self-Evolving Quant AI — v5

**What it is:** A production-ready FastAPI backend that runs a self-evolving trading intelligence:
- Ensemble (LSTM + Transformer + Meta-learner)
- Online learning (incremental updates on stream)
- Full retrain & auto-promotion via Model Registry
- Simple backtest evaluator and metrics
- Logs via Loguru

## Quickstart (Local)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API
- `GET /status` — registry info, readiness
- `POST /predict` — body:
```json
{
  "symbol": "XAUUSD",
  "price": 2350.5,
  "equity": 2000,
  "regime": "trending_up",
  "drift": 0.1,
  "features": {"rsi":55,"macd":0.2,"bb_pos":0.6,"vol_ratio":1.1,"momentum":0.1,"volatility":1.0,"trend":0.2,"sup_dist":0.4,"res_dist":0.7,"tod":13,"sentiment":0.0,"macro_state":1}
}
```
- `POST /train_online?steps=1` — perform small online update
- `POST /retrain` — full mini-retrain, evaluate, and auto-promote if better

## Deploy on Render
- New Web Service → Python
- Build: `pip install -r backend/requirements.txt`
- Start: `bash start.sh`
- Health check: `/status`

## Model Files
Saved under `models/v5/<version>/`. The registry keeps `registry.json` with best_version and metrics.

---

**Note**: This service produces signals; execution on MT5/Deriv happens via your bridge app.
