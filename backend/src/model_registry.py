
from pathlib import Path
import json, shutil, time

class ModelRegistry:
    """
    Simple local registry that keeps best model metadata and versions on disk.
    """
    def __init__(self, root: str = "models/v5"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "registry.json"
        if not self.meta_path.exists():
            self._save({"best_version": None, "versions": {}})

    def _load(self):
        with open(self.meta_path, "r") as f:
            return json.load(f)

    def _save(self, obj):
        with open(self.meta_path, "w") as f:
            json.dump(obj, f, indent=2)

    def create_version_dir(self):
        v = f"v{int(time.time())}"
        d = self.root / v
        d.mkdir(parents=True, exist_ok=True)
        return v, d

    def record_version(self, version: str, metrics: dict, files: dict):
        meta = self._load()
        meta["versions"][version] = {"metrics": metrics, "files": files, "created_at": time.time()}
        # if no best yet, set this
        if meta["best_version"] is None:
            meta["best_version"] = version
        self._save(meta)

    def promote_if_better(self, version: str, new_metrics: dict, cmp_key: str = "sharpe", mdd_key: str = "max_drawdown"):
        meta = self._load()
        best = meta.get("best_version")
        promote = False
        if best is None:
            promote = True
        else:
            best_metrics = meta["versions"][best]["metrics"]
            # better Sharpe and not worse MDD by more than 20%
            if new_metrics.get(cmp_key, 0) > best_metrics.get(cmp_key, 0) and \
               new_metrics.get(mdd_key, 1) <= best_metrics.get(mdd_key, 1) * 1.2:
                promote = True
        if promote:
            meta["best_version"] = version
            self._save(meta)
        return promote, meta["best_version"]

    def get_best_dir(self):
        meta = self._load()
        best = meta.get("best_version")
        if best is None: return None
        return str((self.root / best).resolve())

    def info(self):
        return self._load()
