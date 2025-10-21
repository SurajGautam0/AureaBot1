
import json, time
from pathlib import Path
from loguru import logger

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except Exception:
            pass
        return super().default(obj)

class Telemetry:
    def __init__(self, log_dir: str = "data/logs", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.add(self.log_dir / "backend.log", rotation="5 MB", retention=10, level=level, enqueue=True)
        logger.add(lambda m: print(m, end=""))

    def event(self, type_: str, **kwargs):
        payload = {"ts": time.time(), "type": type_, **kwargs}
        logger.info(json.dumps(payload, cls=NumpyJSONEncoder))

    def metric(self, name: str, value: float, **labels):
        self.event("metric", name=name, value=value, **labels)

    def error(self, message: str, **ctx):
        self.event("error", message=message, **ctx)

    def trade(self, **info):
        self.event("trade", **info)
