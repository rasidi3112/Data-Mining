import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from dmtoolkit.evaluation import (
    bayes_optimal_threshold,
    bootstrap_ci,
    decision_cost,
    expected_calibration_error,
    lift_table,
    nested_cross_validation,
    optimal_threshold,
    reliability_table,
)


def test_bayes_optimal_threshold_formula():
    assert bayes_optimal_threshold(cost_fp=1, cost_fn=1) == pytest.approx(0.5)
    assert bayes_optimal_threshold(cost_fp=1, cost_fn=9) == pytest.approx(0.1)
    assert bayes_optimal_threshold(50, 600, cost_tp=410) == pytest.approx(50 / 240)
    with pytest.raises(ValueError):
        bayes_optimal_threshold(cost_fp=0, cost_fn=1)


def test_decision_cost_counts():
    y = [1, 1, 0, 0]
    pred = [1, 0, 1, 0]  # TP, FN, FP, TN
    assert decision_cost(y, pred, cost_fp=2, cost_fn=5, cost_tp=1, cost_tn=0) == 8


def test_optimal_threshold_close_to_bayes_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(p.size) < p).astype(int)  # p terkalibrasi sempurna
    best, curve = optimal_threshold(y, p, cost_fp=1, cost_fn=4)
    assert best == pytest.approx(bayes_optimal_threshold(1, 4), abs=0.02)
    assert {"threshold", "total_cost", "precision", "recall"} <= set(curve.columns)


def test_ece_near_zero_for_calibrated_and_large_for_miscalibrated():
    rng = np.random.default_rng(1)
    p = rng.random(100_000)
    y = (rng.random(p.size) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    overconfident = np.clip((p - 0.5) * 3 + 0.5, 0, 1)
    assert expected_calibration_error(y, overconfident) > 0.1


def test_reliability_table_counts_and_edge_values():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.95, 0.05, 0.5])
    table = reliability_table(y, p, n_bins=10)
    assert table["count"].sum() == len(y)
    with pytest.raises(ValueError):
        reliability_table([0, 1], [0.2, 1.2])


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    score = y * 0.6 + rng.random(400) * 0.7
    r1 = bootstrap_ci(y, score, roc_auc_score, n_resamples=300, random_state=3)
    r2 = bootstrap_ci(y, score, roc_auc_score, n_resamples=300, random_state=3)
    assert r1.lower <= r1.estimate <= r1.upper
    assert (r1.lower, r1.upper) == (r2.lower, r2.upper)
    assert r1.n_resamples == 300
    assert "CI 95%" in str(r1)


def test_lift_table_properties():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 1000)
    score = y + rng.normal(0, 0.8, 1000)
    table = lift_table(y, score, n_bins=10)
    assert table["n"].sum() == 1000
    assert table["positif"].sum() == y.sum()
    assert table["cumulative_gain"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_lift"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > table["lift"].iloc[-1]


def test_nested_cross_validation_returns_one_row_per_outer_fold():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(0, 0.5, 200) > 0).astype(int))
    search = GridSearchCV(LogisticRegression(max_iter=500), {"C": [0.1, 1.0]},
                          cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc")
    result = nested_cross_validation(search, X, y, StratifiedKFold(4, shuffle=True, random_state=1))
    assert len(result) == 4
    assert result["outer_score"].between(0.5, 1.0).all()
    assert all(p["C"] in (0.1, 1.0) for p in result["best_params"])
