"""Generator dataset sintetis yang deterministik untuk latihan.

Seluruh data di modul ini **disimulasikan**, bukan data nyata. Pola di
dalamnya sengaja ditanamkan agar hasil analisis dapat diverifikasi
(kita tahu "kebenaran" yang seharusnya ditemukan oleh algoritma).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

__all__ = ["make_market_basket", "make_churn_dataset", "make_transaction_anomalies"]


# ---------------------------------------------------------------------------
# 1. Market basket
# ---------------------------------------------------------------------------

_BASE_ITEMS = {
    # item: probabilitas muncul secara independen
    "roti": 0.30, "susu": 0.28, "telur": 0.22, "kopi": 0.25, "teh": 0.18,
    "mie instan": 0.26, "beras": 0.15, "minyak goreng": 0.12, "air mineral": 0.35,
    "sabun": 0.10, "sampo": 0.08, "pasta gigi": 0.09, "biskuit": 0.16,
    "pasta": 0.07, "saus tomat": 0.08, "popok": 0.05, "buah": 0.14,
}

# (pemicu, item yang ikut terbeli, probabilitas tambahan)
_PLANTED_RULES = [
    (("roti",), "selai", 0.55),
    (("kopi",), "gula", 0.50),
    (("teh",), "gula", 0.35),
    (("popok",), "susu bayi", 0.65),
    (("mie instan",), "kerupuk", 0.30),
    (("pasta", "saus tomat"), "keju", 0.60),
    (("sampo",), "sabun", 0.45),
]


def make_market_basket(n_transactions: int = 3000, random_state: int = 42) -> List[List[str]]:
    """Simulasi transaksi minimarket dengan aturan asosiasi yang ditanam.

    Aturan yang ditanam (lihat ``_PLANTED_RULES``), misalnya roti -> selai
    dan {pasta, saus tomat} -> keju, seharusnya muncul dengan lift tinggi.
    """
    rng = np.random.default_rng(random_state)
    items = list(_BASE_ITEMS)
    probs = np.array(list(_BASE_ITEMS.values()))

    transactions: List[List[str]] = []
    for _ in range(n_transactions):
        basket = {item for item, p in zip(items, probs) if rng.random() < p}
        # Pasangan pasta + saus tomat lebih sering dibeli bersama (menu tertentu).
        if "pasta" in basket and rng.random() < 0.5:
            basket.add("saus tomat")
        for triggers, item, p in _PLANTED_RULES:
            if all(t in basket for t in triggers) and rng.random() < p:
                basket.add(item)
        if not basket:
            basket.add(items[int(rng.integers(len(items)))])
        transactions.append(sorted(basket))
    return transactions


# ---------------------------------------------------------------------------
# 2. Churn pelanggan (tabular campuran + missing + drift opsional)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_churn_dataset(
    n_samples: int = 3000,
    random_state: int = 42,
    shift: float = 0.0,
    missing: bool = True,
) -> pd.DataFrame:
    """Simulasi data pelanggan layanan berlangganan beserta label churn.

    Parameters
    ----------
    n_samples : int
        Jumlah pelanggan.
    random_state : int
        Seed agar hasil dapat direproduksi.
    shift : float, default 0.0
        Besar *covariate shift* (0 = sama dengan data training). Nilai > 0
        menyimulasikan kondisi pasar yang berubah: biaya naik, pelanggan baru
        lebih banyak, keluhan meningkat, dan pergeseran metode bayar. Hubungan
        fitur -> churn (konsep) tetap sama.
    missing : bool, default True
        Sisipkan missing values secara acak pada beberapa kolom.

    Returns
    -------
    pandas.DataFrame
        Kolom fitur, ``id_pelanggan`` (bukan fitur!), dan target ``churn``.
    """
    rng = np.random.default_rng(random_state)
    n = n_samples

    plan = rng.choice(["basic", "pro", "enterprise"], size=n, p=[0.55, 0.32, 0.13])
    kontrak = rng.choice(["bulanan", "tahunan", "dua_tahun"], size=n, p=[0.55, 0.30, 0.15])
    pay_p = np.array([0.30, 0.25, 0.30, 0.15])
    if shift:
        pay_p = pay_p + shift * np.array([-0.10, -0.05, 0.20, -0.05])
        pay_p = np.clip(pay_p, 0.01, None)
        pay_p = pay_p / pay_p.sum()
    metode_bayar = rng.choice(["transfer", "kartu_kredit", "e_wallet", "virtual_account"], size=n, p=pay_p)
    region = rng.choice(["barat", "tengah", "timur"], size=n, p=[0.5, 0.3, 0.2])

    tenure_scale = 22.0 * (1 - 0.35 * shift)
    tenure = np.clip(np.round(rng.gamma(shape=1.6, scale=tenure_scale, size=n)) + 1, 1, 72).astype(int)
    tenure = np.where(kontrak == "dua_tahun", np.maximum(tenure, rng.integers(12, 30, size=n)), tenure)

    harga_dasar = pd.Series(plan).map({"basic": 99_000, "pro": 189_000, "enterprise": 349_000}).to_numpy()
    biaya = harga_dasar * (1 + 0.15 * shift) * rng.normal(1.0, 0.08, size=n)
    biaya = np.round(biaya, -3)

    keluhan = rng.poisson(lam=0.8 * (1 + 0.6 * shift), size=n)
    telat_bayar = rng.poisson(lam=np.where(metode_bayar == "transfer", 0.9, 0.4))
    login = rng.poisson(lam=np.clip(18 - 2.5 * keluhan + 0.08 * tenure, 2, None))
    penggunaan_gb = np.round(rng.lognormal(mean=np.log(25) + 0.2 * (plan != "basic"), sigma=0.6, size=n), 1)

    # Model "sebenarnya" (tidak diketahui oleh algoritma). Sengaja memuat
    # efek ambang (threshold), saturasi, dan interaksi yang sulit ditangkap
    # model linear tanpa feature engineering:
    # - risiko tinggi di 6 bulan pertama, efek tenure melandai setelah 24 bulan
    # - keluhan jauh lebih berbahaya pada kontrak bulanan (interaksi)
    # - lonjakan risiko bila login < 8 kali/bulan atau pemakaian < 10 GB
    logit = (
        -1.0
        - 0.07 * np.minimum(tenure, 24)
        + 1.5 * (tenure <= 6)
        + 0.10 * keluhan
        + 0.75 * keluhan * (kontrak == "bulanan")
        + 0.35 * telat_bayar
        + 1.6 * (login < 8)
        + 1.2 * (penggunaan_gb < 10)
        + 0.0000035 * (biaya - harga_dasar)
        + pd.Series(kontrak).map({"bulanan": 0.5, "tahunan": -0.3, "dua_tahun": -1.4}).to_numpy()
        + 0.5 * (metode_bayar == "e_wallet")
        - 0.25 * (plan == "enterprise")
    )
    churn = (rng.random(n) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame({
        "id_pelanggan": [f"CUST-{random_state:03d}-{i:05d}" for i in range(n)],
        "tenure_bulan": tenure,
        "biaya_bulanan": biaya,
        "jumlah_keluhan_3bln": keluhan,
        "telat_bayar_6bln": telat_bayar,
        "login_30hr": login.astype(float),
        "penggunaan_gb": penggunaan_gb,
        "plan": plan,
        "kontrak": kontrak,
        "metode_bayar": metode_bayar,
        "region": region,
        "churn": churn,
    })

    if missing:
        for col, rate in [("penggunaan_gb", 0.05), ("login_30hr", 0.03), ("metode_bayar", 0.02)]:
            idx = rng.choice(n, size=int(rate * n), replace=False)
            df.loc[idx, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 3. Transaksi keuangan dengan anomali
# ---------------------------------------------------------------------------

def make_transaction_anomalies(
    n_normal: int = 3000,
    n_anomalies: int = 60,
    random_state: int = 42,
    return_kind: bool = False,
):
    """Simulasi transaksi kartu dengan dua jenis anomali.

    - **Global**: nilai transaksi ekstrem (mudah dideteksi secara univariat).
    - **Kontekstual/multivariat**: setiap nilai tampak wajar sendiri-sendiri,
      tetapi kombinasinya janggal (mis. dini hari, jauh dari rumah, frekuensi
      tinggi). Jenis ini sulit ditangkap oleh z-score per kolom.

    Returns
    -------
    (X, y) dengan y = 1 untuk anomali. Jika ``return_kind=True``, dikembalikan
    juga Series ``jenis`` bernilai "normal", "global", atau "kontekstual".
    """
    rng = np.random.default_rng(random_state)

    def normal_block(size: int) -> pd.DataFrame:
        jam = np.clip(rng.normal(14, 3.5, size=size), 0, 23.99)
        return pd.DataFrame({
            "nominal_ribu": np.round(rng.lognormal(np.log(150), 0.55, size=size), 1),
            "jam": np.round(jam, 2),
            "jarak_km": np.round(rng.gamma(2.0, 3.0, size=size), 2),
            "transaksi_24jam": rng.poisson(3, size=size).astype(float),
            "merchant_baru": (rng.random(size) < 0.08).astype(float),
        })

    normal = normal_block(n_normal)

    n_global = n_anomalies // 3
    n_context = n_anomalies - n_global

    global_anom = normal_block(n_global)
    global_anom["nominal_ribu"] = np.round(rng.lognormal(np.log(4000), 0.4, size=n_global), 1)

    context = pd.DataFrame({
        "nominal_ribu": np.round(rng.lognormal(np.log(320), 0.35, size=n_context), 1),
        "jam": np.round(rng.uniform(1.0, 4.5, size=n_context), 2),
        "jarak_km": np.round(rng.uniform(15, 30, size=n_context), 2),
        "transaksi_24jam": rng.integers(6, 10, size=n_context).astype(float),
        "merchant_baru": np.ones(n_context),
    })

    X = pd.concat([normal, global_anom, context], ignore_index=True)
    y = pd.Series(np.r_[np.zeros(n_normal), np.ones(n_anomalies)].astype(int), name="anomali")
    kind = pd.Series(["normal"] * n_normal + ["global"] * n_global + ["kontekstual"] * n_context,
                     name="jenis")
    order = rng.permutation(len(X))
    X, y, kind = (obj.iloc[order].reset_index(drop=True) for obj in (X, y, kind))
    return (X, y, kind) if return_kind else (X, y)
