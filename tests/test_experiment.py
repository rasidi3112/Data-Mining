import json

import numpy as np
import pandas as pd
import pytest

from dmtoolkit.experiment import ExperimentTracker, ModelCard, dataframe_fingerprint


def test_fingerprint_deterministic_and_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert dataframe_fingerprint(df) == dataframe_fingerprint(df.copy())
    changed = df.copy()
    changed.loc[0, "a"] = 99
    assert dataframe_fingerprint(df) != dataframe_fingerprint(changed)
    assert dataframe_fingerprint(df) != dataframe_fingerprint(df.rename(columns={"a": "c"}))
    assert dataframe_fingerprint(df) != dataframe_fingerprint(df.iloc[::-1])


def test_tracker_roundtrip(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs" / "exp.jsonl")
    tracker.log_run("lr", params={"C": np.float64(1.0)}, metrics={"auc": np.float32(0.80)})
    tracker.log_run("rf", params={"depth": np.int64(5), "grid": (1, 2)}, metrics={"auc": 0.85})
    frame = tracker.to_frame()
    assert len(frame) == 2
    assert {"params.C", "metrics.auc"} <= set(frame.columns)
    assert tracker.best_run("auc")["name"] == "rf"
    assert tracker.best_run("auc", mode="min")["name"] == "lr"
    with pytest.raises(ValueError):
        tracker.best_run("tidak_ada")
    tracker.clear()
    assert tracker.to_frame().empty


def test_model_card(tmp_path):
    card = ModelCard(
        model_name="churn", version="1.0", problem="prediksi churn", intended_use="retensi",
        metrics={"pr_auc": np.float64(0.7)}, decision_threshold=0.21, limitations=["data sintetis"],
    )
    path = card.save_json(tmp_path / "card.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metrics"]["pr_auc"] == pytest.approx(0.7)
    md = card.to_markdown()
    assert md.startswith("# Model Card: churn")
    assert "0.210" in md and "data sintetis" in md
