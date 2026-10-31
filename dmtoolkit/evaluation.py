"""Evaluasi model tingkat lanjut: keputusan berbasis biaya, kalibrasi,
ketidakpastian (bootstrap), lift/gain table, dan nested cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "bayes_optimal_threshold",
    "decision_cost",
    "threshold_cost_curve",
    "optimal_threshold",
    "reliability_table",
    "expected_calibration_error",
    "BootstrapResult",
    "bootstrap_ci",
    "lift_table",
    "nested_cross_validation",
]


# ---------------------------------------------------------------------------
# Keputusan berbasis biaya (cost-sensitive decision)
# ---------------------------------------------------------------------------

def bayes_optimal_threshold(
    cost_fp: float,
    cost_fn: float,
    cost_tp: float = 0.0,
    cost_tn: float = 0.0,
) -> float:
    """Threshold optimal secara teori untuk probabilitas yang **terkalibrasi**.

    Prediksi positif dipilih jika biaya harapannya lebih kecil:

        p * C_TP + (1 - p) * C_FP  <=  p * C_FN + (1 - p) * C_TN

    sehingga

        p >= t* = (C_FP - C_TN) / ((C_FP - C_TN) + (C_FN - C_TP))

    Syarat logis: C_FP > C_TN dan C_FN > C_TP (salah prediksi lebih mahal
    daripada prediksi benar).
    """
    a = cost_fp - cost_tn
    b = cost_fn - cost_tp
    if a <= 0 or b <= 0:
        raise ValueError("Syarat: cost_fp > cost_tn dan cost_fn > cost_tp.")
    return a / (a + b)


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def decision_cost(
    y_true,
    y_pred,
    cost_fp: float,
    cost_fn: float,
    cost_tp: float = 0.0,
    cost_tn: float = 0.0,
) -> float:
    """Total biaya dari sekumpulan keputusan biner berdasarkan matriks biaya."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    return float(tp * cost_tp + fp * cost_fp + fn * cost_fn + tn * cost_tn)


def threshold_cost_curve(
    y_true,
    y_score,
    cost_fp: float,
    cost_fn: float,
    cost_tp: float = 0.0,
    cost_tn: float = 0.0,
    thresholds: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Hitung biaya, precision, dan recall untuk setiap threshold.

    Sampel diprediksi positif jika ``y_score >= threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 4)

    rows = []
    n = len(y_true)
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
        total = tp * cost_tp + fp * cost_fp + fn * cost_fn + tn * cost_tn
        rows.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "flag_rate": (tp + fp) / n if n else np.nan,
            "total_cost": float(total),
            "cost_per_sample": float(total / n) if n else np.nan,
        })
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true,
    y_score,
    cost_fp: float,
    cost_fn: float,
    cost_tp: float = 0.0,
    cost_tn: float = 0.0,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, pd.DataFrame]:
    """Threshold empiris dengan total biaya terendah.

    Jika beberapa threshold memiliki biaya sama, dipilih yang paling dekat ke
    0.5 agar keputusan tidak terlalu ekstrem.

    Returns
    -------
    (threshold_terbaik, tabel_kurva_biaya)
    """
    curve = threshold_cost_curve(y_true, y_score, cost_fp, cost_fn, cost_tp, cost_tn, thresholds)
    best_cost = curve["total_cost"].min()
    ties = curve[np.isclose(curve["total_cost"], best_cost)]
    best = ties.iloc[(ties["threshold"] - 0.5).abs().argmin()]
    return float(best["threshold"]), curve


# ---------------------------------------------------------------------------
# Kalibrasi probabilitas
# ---------------------------------------------------------------------------

def _bin_edges(y_prob: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    raise ValueError("strategy harus 'uniform' atau 'quantile'.")


def reliability_table(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    """Tabel reliabilitas: rata-rata prediksi vs frekuensi positif per bin."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true dan y_prob harus berukuran sama.")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob harus berada pada rentang [0, 1].")

    edges = _bin_edges(y_prob, n_bins, strategy)
    # Bin terakhir inklusif di kanan agar p = 1.0 tetap masuk.
    bin_ids = np.clip(np.searchsorted(edges[1:-1], y_prob, side="right"), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        rows.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": count,
            "mean_predicted": mean_pred,
            "fraction_positive": frac_pos,
            "abs_gap": abs(frac_pos - mean_pred),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected Calibration Error (ECE) untuk klasifikasi biner.

        ECE = sum_b (n_b / N) * | frac_pos_b - mean_pred_b |

    ECE = 0 berarti probabilitas sempurna terkalibrasi pada resolusi bin tsb.
    """
    table = reliability_table(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if table.empty:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    return float((weights * table["abs_gap"]).sum())


# ---------------------------------------------------------------------------
# Ketidakpastian metrik: bootstrap
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Ringkasan selang kepercayaan bootstrap persentil."""

    estimate: float
    lower: float
    upper: float
    std: float
    confidence: float
    n_resamples: int

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return (f"{self.estimate:.4f} (CI {pct}%: {self.lower:.4f} - {self.upper:.4f}, "
                f"SE~{self.std:.4f}, B={self.n_resamples})")


def bootstrap_ci(
    y_true,
    y_pred,
    metric: Callable,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    stratify: bool = True,
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """Selang kepercayaan metrik dengan bootstrap persentil.

    Data test di-resample dengan pengembalian sebanyak ``n_resamples`` kali,
    metrik dihitung ulang, lalu diambil persentil ``(1-confidence)/2`` dan
    ``1-(1-confidence)/2``. Dengan ``stratify=True`` jumlah sampel per kelas
    dipertahankan sehingga metrik seperti ROC-AUC selalu terdefinisi.

    ``y_pred`` dapat berupa label prediksi atau skor/probabilitas, sesuai
    kebutuhan ``metric(y_true, y_pred)``.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence harus pada rentang (0, 1).")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true dan y_pred harus berukuran sama.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)] if stratify else [np.arange(n)]

    scores = []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        try:
            scores.append(float(metric(y_true[idx], y_pred[idx])))
        except ValueError:
            # Contoh: resample tanpa stratifikasi yang hanya berisi satu kelas.
            continue
    if not scores:
        raise RuntimeError("Semua resample gagal dihitung; coba stratify=True.")

    scores_arr = np.asarray(scores)
    alpha = 1 - confidence
    lower, upper = np.percentile(scores_arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        estimate=float(metric(y_true, y_pred)),
        lower=float(lower),
        upper=float(upper),
        std=float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else 0.0,
        confidence=confidence,
        n_resamples=len(scores_arr),
    )


# ---------------------------------------------------------------------------
# Lift / gain table (pemasaran, prioritisasi)
# ---------------------------------------------------------------------------

def lift_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Tabel gain & lift per kelompok (default desil) berdasarkan urutan skor.

    - ``capture_rate``: porsi seluruh positif yang tertangkap di kelompok itu
    - ``cumulative_gain``: porsi positif kumulatif jika menyasar kelompok 1..k
    - ``lift``: rasio tingkat positif kelompok terhadap tingkat positif global
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < n_bins:
        raise ValueError("Jumlah sampel harus >= n_bins.")
    total_pos = y_true.sum()
    if total_pos == 0:
        raise ValueError("y_true tidak memiliki kelas positif.")
    base_rate = total_pos / len(y_true)

    # mergesort = stabil, sehingga hasil deterministik saat ada skor kembar.
    order = np.argsort(-y_score, kind="mergesort")
    rows = []
    cum_pos = 0
    cum_n = 0
    for k, chunk in enumerate(np.array_split(order, n_bins), start=1):
        pos = int(y_true[chunk].sum())
        cum_pos += pos
        cum_n += len(chunk)
        rate = pos / len(chunk)
        rows.append({
            "kelompok": k,
            "n": len(chunk),
            "skor_min": float(y_score[chunk].min()),
            "skor_max": float(y_score[chunk].max()),
            "positif": pos,
            "tingkat_positif": rate,
            "capture_rate": pos / total_pos,
            "cumulative_gain": cum_pos / total_pos,
            "lift": rate / base_rate,
            "cumulative_lift": (cum_pos / cum_n) / base_rate,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------------

def _take(data, idx):
    return data.iloc[idx] if hasattr(data, "iloc") else np.asarray(data)[idx]


def nested_cross_validation(search, X, y, outer_cv, scoring=None) -> pd.DataFrame:
    """Nested CV: estimasi performa *seluruh prosedur* tuning secara jujur.

    Parameters
    ----------
    search : GridSearchCV / RandomizedSearchCV (belum di-fit)
        Objek pencarian hyperparameter dengan ``cv`` sebagai *inner loop*.
    X, y : data training (DataFrame/array)
    outer_cv : splitter scikit-learn untuk *outer loop*
    scoring : str atau callable, optional
        Metrik untuk outer loop. Default: sama dengan ``search.scoring``.

    Returns
    -------
    pandas.DataFrame
        Satu baris per outer fold: skor outer, skor terbaik inner, parameter.
    """
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    scoring = scoring if scoring is not None else getattr(search, "scoring", None)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        fitted = clone(search).fit(_take(X, train_idx), _take(y, train_idx))
        scorer = check_scoring(fitted.best_estimator_, scoring=scoring)
        outer_score = scorer(fitted.best_estimator_, _take(X, test_idx), _take(y, test_idx))
        rows.append({
            "fold": fold,
            "inner_best_score": float(fitted.best_score_),
            "outer_score": float(outer_score),
            "best_params": fitted.best_params_,
        })
    return pd.DataFrame(rows)
