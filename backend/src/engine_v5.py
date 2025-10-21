
import numpy as np, torch, torch.nn as nn, os, json, time
from collections import deque
from typing import Dict, List, Tuple
from .model_registry import ModelRegistry
from .evaluator import simple_backtest, metrics_from_equity

# --- Models ---
class LSTM(nn.Module):
    def __init__(self, input_size=12, hidden=48, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, 3)   # dir, conf, mag
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:,-1,:])

class Transf(nn.Module):
    def __init__(self, input_size=12, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.embed = nn.Linear(input_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Linear(d_model, 3)
    def forward(self, x):
        x = self.embed(x).transpose(0,1) # (seq,batch,feat)
        z = self.enc(x)
        return self.fc(z[-1])

class Meta(nn.Module):
    def __init__(self, in_dim=3*2+6): # two models *3 + 6 context
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, mp, ctx):
        x = torch.cat([mp, ctx], dim=-1)
        return torch.tanh(self.net(x))

# --- Engine v5 ---
class CognitiveEngineV5:
    """
    Self-evolving ensemble with online-learning and automatic model promotion.
    """
    def __init__(self, config: Dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = self.cfg.get("input_size", 12)
        self.seq_len = self.cfg.get("sequence_length", 48)

        self.lstm = LSTM(self.input_size, hidden=self.cfg.get("lstm_hidden",48)).to(self.device)
        self.trf = Transf(self.input_size, d_model=self.cfg.get("tr_d_model",64)).to(self.device)
        self.meta = Meta(in_dim=6+6).to(self.device)

        self.hist_feats = deque(maxlen=5000)
        self.hist_prices = deque(maxlen=5000)
        self.seq_buf = deque(maxlen=self.seq_len)

        self.registry = ModelRegistry(self.cfg.get("registry_root","models/v5"))
        self.model_dir = self.registry.get_best_dir() or self.cfg.get("model_dir","models/v5/bootstrap")
        os.makedirs(self.model_dir, exist_ok=True)

        self.pred_history = deque(maxlen=2000)

    def _feat_vec(self, f: Dict, price: float):
        return np.array([
            f.get("rsi",50)/100.0,
            f.get("macd",0)/10.0,
            f.get("bb_pos",0.5),
            f.get("vol_ratio",1.0),
            f.get("momentum",0)/10.0,
            f.get("volatility",1.0),
            f.get("trend",0.0),
            f.get("sup_dist",0.5),
            f.get("res_dist",0.5),
            f.get("tod",12)/24.0,
            f.get("sentiment",0.0),          # new
            f.get("macro_state",0.0)         # new
        ], dtype=np.float32)

    def add_tick(self, features: Dict, price: float):
        v = self._feat_vec(features, price)
        self.seq_buf.append(v)
        self.hist_feats.append(v)
        self.hist_prices.append(float(price))

    @torch.no_grad()
    def predict(self, features: Dict, price: float) -> Dict:
        self.add_tick(features, price)
        if len(self.seq_buf) < max(16, self.seq_len//2):
            return {"direction":"HOLD","confidence":0.55,"magnitude":0.0,"meta":{"booting":True}}

        seq = torch.tensor([np.array(self.seq_buf)], dtype=torch.float32, device=self.device)
        p_l = self.lstm(seq)
        p_t = self.trf(seq)

        # context: vol, data_richness, time, sentiment, drift proxy, regime proxy
        prices = np.array(self.hist_prices)[-32:] if len(self.hist_prices)>=32 else np.array(self.hist_prices)
        vol = float(np.std(np.diff(prices))) if len(prices)>2 else 1.0
        ctx = torch.tensor([[
            vol, min(1.0, len(self.seq_buf)/self.seq_len),
            (time.time()%86400.0)/86400.0,
            features.get("sentiment",0.0),
            features.get("drift",0.0),
            1.0 if features.get("regime","neutral") in ("trending_up","trending_down") else 0.0
        ]], dtype=torch.float32, device=self.device)

        mp = torch.cat([p_l, p_t], dim=-1)
        out = self.meta(mp, ctx).squeeze(0).detach().cpu().numpy()  # [-1,1]^3

        dir_score, conf_raw, mag_raw = float(out[0]), float(abs(out[1])), float(abs(out[2]))
        direction = "BUY" if dir_score>0 else "SELL"
        confidence = float(np.clip(0.70 + 0.25*conf_raw, 0.0, 1.0))
        magnitude = float(mag_raw)

        self.pred_history.append({"side": 1 if direction=="BUY" else -1, "conf": confidence})
        return {"direction":direction,"confidence":confidence,"magnitude":magnitude,
                "meta":{"vol":vol,"ctx_len":len(self.seq_buf)}}

    def online_train(self, steps:int=1, lr:float=5e-4):
        """
        Small online updates using recent stream. Targets = sign of short-term delta.
        """
        n = min(512, len(self.hist_feats)-1)
        if n < self.seq_len+2: return {"ok":False,"msg":"not_enough_data"}
        feats = np.array(list(self.hist_feats)[-n:], dtype=np.float32)
        prices = np.array(list(self.hist_prices)[-n:], dtype=np.float32)
        deltas = np.sign(np.diff(prices))
        deltas[deltas==0]=1.0

        X, y = [], []
        for i in range(self.seq_len, n-1):
            X.append(feats[i-self.seq_len:i])
            y.append([deltas[i-1], 1.0, abs(deltas[i-1])])
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)

        crit = nn.MSELoss()
        for model in (self.lstm, self.trf):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(steps):
                model.train(); opt.zero_grad()
                out = model(X)
                loss = crit(out, y)
                loss.backward(); opt.step()
        return {"ok":True, "samples": int(X.size(0))}

    def retrain_and_evaluate(self, epochs:int=6, lr:float=1e-3):
        """
        Full mini-retrain on all stored data; evaluate; register & possibly promote.
        """
        n = len(self.hist_feats)
        if n < self.seq_len + 50:
            return {"ok":False,"msg":"need_more_history"}

        feats = np.array(list(self.hist_feats), dtype=np.float32)
        prices = np.array(list(self.hist_prices), dtype=np.float32)
        deltas = np.sign(np.diff(prices)); deltas[deltas==0]=1.0

        X, y = [], []
        for i in range(self.seq_len, n-1):
            X.append(feats[i-self.seq_len:i])
            y.append([deltas[i-1], 1.0, abs(deltas[i-1])])
        X = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.array(y), dtype=torch.float32, device=self.device)

        crit = nn.MSELoss()
        opt_l = torch.optim.Adam(self.lstm.parameters(), lr=lr)
        opt_t = torch.optim.Adam(self.trf.parameters(), lr=lr)
        self.lstm.train(); self.trf.train()
        for e in range(epochs):
            for model,opt in ((self.lstm,opt_l),(self.trf,opt_t)):
                opt.zero_grad(); out = model(X); loss = crit(out,y); loss.backward(); opt.step()

        # Evaluate via simple backtest on the last chunk
        with torch.no_grad():
            preds = []
            for i in range(self.seq_len, n):
                seq = torch.tensor([feats[i-self.seq_len:i]], dtype=torch.float32, device=self.device)
                pl = self.lstm(seq)
                pt = self.trf(seq)
                ctx = torch.tensor([[0.5,1.0,0.5,0.0,0.0,0.5]], dtype=torch.float32, device=self.device)
                out = self.meta(torch.cat([pl,pt],dim=-1), ctx).squeeze(0).cpu().numpy()
                side = 1 if out[0]>0 else -1
                conf = float(np.clip(0.70+0.25*abs(out[1]),0,1))
                preds.append({"side":side,"conf":conf})
        eq = simple_backtest(preds, prices[-len(preds):])
        mets = metrics_from_equity(eq)

        # Save checkpoint
        version, vdir = self.registry.create_version_dir()
        torch.save(self.lstm.state_dict(), str((vdir/"lstm.pth").resolve()))
        torch.save(self.trf.state_dict(), str((vdir/"transformer.pth").resolve()))
        torch.save(self.meta.state_dict(), str((vdir/"meta.pth").resolve()))
        files = {"lstm":"lstm.pth","transformer":"transformer.pth","meta":"meta.pth"}
        self.registry.record_version(version, mets, files)

        promoted, best = self.registry.promote_if_better(version, mets)
        return {"ok":True, "version":version, "metrics":mets, "promoted":promoted, "best_version":best}

    def load_best(self):
        best_dir = self.registry.get_best_dir()
        if not best_dir: return False
        try:
            self.lstm.load_state_dict(torch.load(os.path.join(best_dir,"lstm.pth"), map_location=self.device))
            self.trf.load_state_dict(torch.load(os.path.join(best_dir,"transformer.pth"), map_location=self.device))
            self.meta.load_state_dict(torch.load(os.path.join(best_dir,"meta.pth"), map_location=self.device))
            self.lstm.to(self.device); self.trf.to(self.device); self.meta.to(self.device)
            return True
        except Exception as e:
            print("load_best failed:", e)
            return False

    def registry_info(self):
        return self.registry.info()
