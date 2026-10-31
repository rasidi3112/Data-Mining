"""Monitoring data drift setelah model dipakai (post-deployment).

Population Stability Index (PSI) membandingkan distribusi data referensi
(biasanya data training) dengan data terbaru:

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

dengan e_i = proporsi data referensi di bin i dan a_i = proporsi data terbaru
di bin i. Nilai yang hilang (NaN) diperlakukan sebagai bin tersendiri karena
perubahan tingkat missing juga merupakan sinyal drift.

Ambang interpretasi yang lazim dipakai dalam praktik (*rule of thumb* dari
dunia credit scoring, bukan standar formal):

- PSI < 0.10        : stabil
- 0.10 <= PSI < 0.25 : perlu dipantau
- PSI >= 0.25       : drift signifikan, evaluasi ulang model
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

__all__ = ["population_stability_index", "psi_status", "drift_report"]

_MISSING = "__missing__"


def _is_categorical(series: pd.Series) -> bool:
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray, eps: float) -> float:
    e = np.clip(expected, eps, None)
    a = np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def population_stability_index(
    expected,
    actual,
    n_bins: int = 10,
    categorical: Optional[bool] = None,
    eps: float = 1e-4,
) -> float:
    """Hitung PSI antara data referensi (``expected``) dan data baru (``actual``).

    Untuk fitur numerik, batas bin diambil dari kuantil data referensi
    sehingga tiap bin referensi berisi porsi data yang kurang lebih sama.
    Untuk fitur kategorikal, tiap kategori menjadi satu bin.
    """
    expected = pd.Series(expected).reset_index(drop=True)
    actual = pd.Series(actual).reset_index(drop=True)
    if expected.empty or actual.empty:
        raise ValueError("expected dan actual tidak boleh kosong.")
    if categorical is None:
        categorical = _is_categorical(expected)

    if categorical:
        e_vals = expected.astype(object).where(expected.notna(), _MISSING).astype(str)
        a_vals = actual.astype(object).where(actual.notna(), _MISSING).astype(str)
        categories = sorted(set(e_vals) | set(a_vals))
        e_prop = e_vals.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
        a_prop = a_vals.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
        return _psi_from_proportions(e_prop, a_prop, eps)

    e_num = pd.to_numeric(expected, errors="raise").astype(float)
    a_num = pd.to_numeric(actual, errors="raise").astype(float)
    e_valid = e_num.dropna().to_numpy()
    if e_valid.size == 0:
        raise ValueError("Data referensi numerik seluruhnya NaN.")

    inner_edges = np.unique(np.quantile(e_valid, np.linspace(0, 1, n_bins + 1))[1:-1])
    edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))

    def proportions(values: pd.Series) -> np.ndarray:
        valid = values.dropna().to_numpy()
        counts, _ = np.histogram(valid, bins=edges)
        missing = values.isna().sum()
        return np.append(counts, missing) / len(values)

    return _psi_from_proportions(proportions(e_num), proportions(a_num), eps)


def psi_status(psi: float) -> str:
    """Label interpretasi PSI berdasarkan ambang praktis 0.10 / 0.25."""
    if np.isnan(psi):
        return "tidak diketahui"
    if psi < 0.10:
        return "stabil"
    if psi < 0.25:
        return "perlu dipantau"
    return "drift signifikan"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Laporan drift per kolom: PSI, uji Kolmogorov-Smirnov (numerik), missing rate.

    Uji KS (dua sampel, metode asimtotik) memerlukan SciPy; jika tidak tersedia, kolom KS bernilai NaN.
    Catatan: pada sampel besar p-value KS hampir selalu kecil walau
    perbedaannya tidak bermakna secara praktis, sehingga PSI (ukuran efek)
    lebih cocok dijadikan acuan utama.
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:  # pragma: no cover - SciPy terpasang bersama scikit-learn
        ks_2samp = None

    columns = list(columns) if columns is not None else [c for c in reference.columns if c in current.columns]
    rows = []
    for col in columns:
        ref, cur = reference[col], current[col]
        categorical = _is_categorical(ref)
        psi = population_stability_index(ref, cur, n_bins=n_bins, categorical=categorical)

        ks_stat = ks_p = np.nan
        if not categorical and ks_2samp is not None:
            r, c = ref.dropna(), cur.dropna()
            if len(r) and len(c):
                # Metode asimtotik: stabil untuk sampel besar & data diskret (banyak nilai kembar).
                result = ks_2samp(r, c, method="asymp")
                ks_stat, ks_p = float(result.statistic), float(result.pvalue)

        rows.append({
            "kolom": col,
            "tipe": "kategorikal" if categorical else "numerik",
            "psi": psi,
            "status": psi_status(psi),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "missing_ref": float(ref.isna().mean()),
            "missing_cur": float(cur.isna().mean()),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
