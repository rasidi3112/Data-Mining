"""Reproducibility & dokumentasi: pencatatan eksperimen, sidik jari data,
dan model card sederhana.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "dataframe_fingerprint",
    "environment_info",
    "ExperimentTracker",
    "ModelCard",
]

PathLike = Union[str, Path]


def _to_jsonable(obj: Any) -> Any:
    """Konversi tipe numpy/pandas/Path ke tipe JSON standar."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Hash SHA-256 dari isi, indeks, dan nama kolom DataFrame.

    Dua DataFrame dengan isi identik menghasilkan hash yang sama; perubahan
    sekecil apa pun (satu sel, urutan baris, nama kolom) mengubah hash.
    Gunakan untuk memastikan eksperimen dijalankan pada versi data yang sama.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode("utf-8"))
    hasher.update("|".join(map(str, df.dtypes)).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return hasher.hexdigest()


def environment_info(packages: Optional[List[str]] = None) -> Dict[str, str]:
    """Versi Python, OS, dan library utama (untuk bagian reproduksi laporan)."""
    packages = packages or ["numpy", "pandas", "sklearn", "scipy", "matplotlib"]
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in packages:
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = "tidak terpasang"
    return info


class ExperimentTracker:
    """Pencatat eksperimen ringan berbasis file JSON Lines.

    Setiap ``log_run`` menambah satu baris JSON ke file, sehingga riwayat
    eksperimen tidak pernah tertimpa dan mudah dibandingkan.

    Contoh
    ------
    >>> tracker = ExperimentTracker("artifacts/experiments.jsonl")
    >>> tracker.log_run("rf_v1", params={"n_estimators": 200}, metrics={"pr_auc": 0.71})
    >>> tracker.to_frame()
    """

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, Any]] = None,
        data_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "run_id": f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
            "name": name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": params or {},
            "metrics": metrics or {},
            "tags": tags or {},
            "data_fingerprint": data_fingerprint,
            "environment": environment_info(),
        }
        record = _to_jsonable(record)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def runs(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def to_frame(self) -> pd.DataFrame:
        """Riwayat eksperimen sebagai DataFrame datar (``params.*``, ``metrics.*``)."""
        runs = self.runs()
        if not runs:
            return pd.DataFrame()
        flat = pd.json_normalize(runs, sep=".")
        env_cols = [c for c in flat.columns if c.startswith("environment.")]
        return flat.drop(columns=env_cols)

    def best_run(self, metric: str, mode: str = "max") -> Dict[str, Any]:
        if mode not in {"max", "min"}:
            raise ValueError("mode harus 'max' atau 'min'.")
        candidates = [r for r in self.runs() if metric in r.get("metrics", {})]
        if not candidates:
            raise ValueError(f"Belum ada run yang mencatat metrik '{metric}'.")
        chooser = max if mode == "max" else min
        return chooser(candidates, key=lambda r: r["metrics"][metric])

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


@dataclass
class ModelCard:
    """Model card ringkas: dokumentasi standar sebuah model sebelum dipakai."""

    model_name: str
    version: str
    problem: str
    intended_use: str
    training_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    decision_threshold: Optional[float] = None
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=environment_info)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))

    def save_json(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        def bullet_dict(d: Dict[str, Any]) -> str:
            return "\n".join(f"- **{k}**: {v}" for k, v in d.items()) or "- (kosong)"

        def bullet_list(items: List[str]) -> str:
            return "\n".join(f"- {x}" for x in items) or "- (kosong)"

        threshold = "-" if self.decision_threshold is None else f"{self.decision_threshold:.3f}"
        return "\n".join([
            f"# Model Card: {self.model_name} (v{self.version})",
            "",
            f"**Masalah:** {self.problem}",
            "",
            f"**Penggunaan yang dimaksud:** {self.intended_use}",
            "",
            f"**Threshold keputusan:** {threshold}",
            "",
            "## Data Training", bullet_dict(self.training_data), "",
            "## Metrik", bullet_dict(self.metrics), "",
            "## Limitasi", bullet_list(self.limitations), "",
            "## Pertimbangan Etis", bullet_list(self.ethical_considerations), "",
            "## Lingkungan", bullet_dict(self.environment),
        ])
