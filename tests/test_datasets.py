import pandas as pd

from dmtoolkit.datasets import make_churn_dataset, make_market_basket, make_transaction_anomalies


def test_market_basket_reproducible_and_nonempty():
    a = make_market_basket(500, random_state=1)
    b = make_market_basket(500, random_state=1)
    assert a == b
    assert len(a) == 500 and all(len(t) > 0 for t in a)


def test_churn_dataset_structure():
    df = make_churn_dataset(2000, random_state=0)
    assert df.shape == (2000, 12)
    assert df["id_pelanggan"].is_unique
    assert set(df["churn"].unique()) == {0, 1}
    assert 0.15 < df["churn"].mean() < 0.40
    assert df["penggunaan_gb"].isna().sum() > 0
    assert make_churn_dataset(2000, random_state=0, missing=False).isna().sum().sum() == 0
    pd.testing.assert_frame_equal(df, make_churn_dataset(2000, random_state=0))


def test_churn_shift_changes_distribution():
    base = make_churn_dataset(4000, random_state=0)
    shifted = make_churn_dataset(4000, random_state=0, shift=1.0)
    assert shifted["biaya_bulanan"].mean() > base["biaya_bulanan"].mean() * 1.1
    assert shifted["tenure_bulan"].mean() < base["tenure_bulan"].mean()


def test_transaction_anomalies():
    X, y = make_transaction_anomalies(n_normal=1000, n_anomalies=30, random_state=0)
    assert len(X) == len(y) == 1030
    assert y.sum() == 30
    assert not X.isna().any().any()


def test_transaction_anomalies_kind_labels():
    X, y, kind = make_transaction_anomalies(n_normal=500, n_anomalies=30, random_state=0, return_kind=True)
    assert kind.value_counts().to_dict() == {"normal": 500, "kontekstual": 20, "global": 10}
    assert ((kind != "normal").astype(int) == y).all()
