import numpy as np
import pandas as pd
import pytest

from dmtoolkit.datasets import make_churn_dataset
from dmtoolkit.drift import drift_report, population_stability_index, psi_status


def test_psi_identical_is_zero_and_nonnegative():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-12)
    assert population_stability_index(x, rng.normal(size=5000)) >= 0


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    assert population_stability_index(ref, rng.normal(0, 1, 5000)) < 0.10
    assert population_stability_index(ref, rng.normal(1.0, 1, 5000)) > 0.25


def test_psi_categorical_and_missing():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 90 + ["b"] * 10)
    assert population_stability_index(ref, cur) > 0.25
    ref_num = pd.Series(np.arange(100, dtype=float))
    cur_num = ref_num.copy()
    cur_num[:40] = np.nan  # hanya tingkat missing yang berubah
    assert population_stability_index(ref_num, cur_num) > 0.10


@pytest.mark.parametrize("value,label", [(0.05, "stabil"), (0.15, "perlu dipantau"), (0.4, "drift signifikan")])
def test_psi_status(value, label):
    assert psi_status(value) == label


def test_drift_report_on_shifted_churn_data():
    ref = make_churn_dataset(3000, random_state=1).drop(columns=["id_pelanggan", "churn"])
    same = make_churn_dataset(3000, random_state=2).drop(columns=["id_pelanggan", "churn"])
    shifted = make_churn_dataset(3000, random_state=3, shift=1.0).drop(columns=["id_pelanggan", "churn"])

    stable = drift_report(ref, same)
    assert (stable["status"] == "stabil").all()

    report = drift_report(ref, shifted).set_index("kolom")
    assert report.loc["biaya_bulanan", "status"] == "drift signifikan"
    assert report.loc["metode_bayar", "tipe"] == "kategorikal"
    assert np.isnan(report.loc["metode_bayar", "ks_statistic"])
    assert report.loc["tenure_bulan", "ks_pvalue"] < 0.01
