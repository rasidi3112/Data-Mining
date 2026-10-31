"""Association rule mining (market basket analysis) dengan algoritma Apriori.

Implementasi ini sengaja ditulis dari nol (tanpa library tambahan seperti
mlxtend) agar setiap langkah algoritma bisa dipelajari dan diuji.

Definisi metrik untuk aturan A -> C, dengan N = jumlah transaksi:

- support(X)      = jumlah transaksi yang memuat X / N
- confidence(A->C) = support(A u C) / support(A)
- lift(A->C)       = confidence(A->C) / support(C)
- leverage(A->C)   = support(A u C) - support(A) * support(C)
- conviction(A->C) = (1 - support(C)) / (1 - confidence(A->C))
- Zhang's metric   = leverage / max(support(A u C) * (1 - support(A)),
                                    support(A) * (support(C) - support(A u C)))

Interpretasi singkat:
- lift > 1   : A dan C muncul bersama lebih sering daripada jika independen
- lift = 1   : A dan C independen
- lift < 1   : A dan C cenderung saling menggantikan
- Zhang's metric berada di rentang [-1, 1]; nilai positif = asosiasi positif.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "encode_transactions",
    "itemset_support",
    "apriori",
    "association_rules",
    "format_itemset",
]

_EPS = 1e-12


def encode_transactions(transactions: Iterable[Iterable]) -> pd.DataFrame:
    """Ubah daftar transaksi menjadi matriks one-hot boolean.

    Parameters
    ----------
    transactions : iterable of iterable
        Contoh: ``[["roti", "selai"], ["kopi"], ["roti", "kopi", "gula"]]``.
        Item duplikat di dalam satu transaksi dihitung satu kali.

    Returns
    -------
    pandas.DataFrame
        Baris = transaksi, kolom = item (urut alfabet), nilai = bool.
    """
    baskets = [set(t) for t in transactions]
    items = sorted(set().union(*baskets), key=str) if baskets else []
    position = {item: j for j, item in enumerate(items)}

    matrix = np.zeros((len(baskets), len(items)), dtype=bool)
    for i, basket in enumerate(baskets):
        for item in basket:
            matrix[i, position[item]] = True
    return pd.DataFrame(matrix, columns=items)


def _as_bool_matrix(basket: pd.DataFrame) -> np.ndarray:
    if not isinstance(basket, pd.DataFrame):
        raise TypeError("basket harus berupa pandas.DataFrame one-hot (hasil encode_transactions).")
    values = basket.to_numpy()
    if values.dtype != bool:
        unique = pd.unique(values.ravel())
        if not set(unique.tolist()) <= {0, 1, True, False}:
            raise ValueError("basket hanya boleh berisi nilai 0/1 atau True/False.")
        values = values.astype(bool)
    return values


def itemset_support(basket: pd.DataFrame, items: Sequence) -> float:
    """Hitung support satu itemset secara langsung (brute force).

    Berguna sebagai pembanding untuk memverifikasi hasil :func:`apriori`.
    """
    matrix = _as_bool_matrix(basket)
    if len(items) == 0:
        return 1.0
    idx = [basket.columns.get_loc(item) for item in items]
    return float(matrix[:, idx].all(axis=1).mean())


def _generate_candidates(previous: List[Tuple[int, ...]], k: int) -> List[Tuple[int, ...]]:
    """Langkah *join* + *prune* Apriori.

    ``previous`` berisi itemset frequent berukuran k-1 (tuple indeks terurut).
    Dua itemset digabung jika (k-2) item pertamanya sama. Kandidat dibuang
    jika ada subset berukuran k-1 yang tidak frequent (sifat anti-monoton /
    downward closure: subset dari itemset frequent pasti frequent).
    """
    previous_sorted = sorted(previous)
    previous_set = set(previous_sorted)
    candidates: List[Tuple[int, ...]] = []

    for i in range(len(previous_sorted)):
        a = previous_sorted[i]
        for j in range(i + 1, len(previous_sorted)):
            b = previous_sorted[j]
            if a[:-1] != b[:-1]:
                # Karena terurut leksikografis, prefix yang sama selalu berdampingan.
                break
            candidate = a + (b[-1],)
            if all(sub in previous_set for sub in combinations(candidate, k - 1)):
                candidates.append(candidate)
    return candidates


def apriori(
    basket: pd.DataFrame,
    min_support: float = 0.05,
    max_len: Optional[int] = None,
) -> pd.DataFrame:
    """Cari semua frequent itemset dengan algoritma Apriori (level-wise).

    Parameters
    ----------
    basket : pandas.DataFrame
        Matriks one-hot transaksi x item.
    min_support : float, default 0.05
        Ambang support minimum, 0 < min_support <= 1.
    max_len : int, optional
        Panjang itemset maksimum. ``None`` berarti tanpa batas.

    Returns
    -------
    pandas.DataFrame
        Kolom ``itemsets`` (frozenset), ``support``, ``length``; diurutkan
        dari support terbesar.
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support harus berada pada rentang (0, 1].")
    if max_len is not None and max_len < 1:
        raise ValueError("max_len minimal 1.")

    matrix = _as_bool_matrix(basket)
    n_transactions = matrix.shape[0]
    columns = np.asarray(basket.columns, dtype=object)
    empty = pd.DataFrame({"itemsets": pd.Series(dtype=object),
                          "support": pd.Series(dtype=float),
                          "length": pd.Series(dtype=int)})
    if n_transactions == 0 or matrix.shape[1] == 0:
        return empty

    # Bandingkan dalam bentuk hitungan (count) agar bebas galat pembulatan float.
    min_count = min_support * n_transactions - 1e-9

    counts_1 = matrix.sum(axis=0)
    current = [(j,) for j in np.flatnonzero(counts_1 >= min_count)]
    frequent: Dict[Tuple[int, ...], int] = {c: int(counts_1[c[0]]) for c in current}

    k = 1
    while current and (max_len is None or k < max_len):
        k += 1
        candidates = _generate_candidates(current, k)
        if not candidates:
            break
        current = []
        for candidate in candidates:
            count = int(matrix[:, candidate].all(axis=1).sum())
            if count >= min_count:
                frequent[candidate] = count
                current.append(candidate)

    if not frequent:
        return empty

    result = pd.DataFrame({
        "itemsets": [frozenset(columns[list(idx)].tolist()) for idx in frequent],
        "support": [count / n_transactions for count in frequent.values()],
        "length": [len(idx) for idx in frequent],
    })
    return result.sort_values(["support", "length"], ascending=[False, True]).reset_index(drop=True)


_RULE_METRICS = ("support", "confidence", "lift", "leverage", "conviction", "zhangs_metric")


def association_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.5,
) -> pd.DataFrame:
    """Bangkitkan aturan asosiasi A -> C dari hasil :func:`apriori`.

    Parameters
    ----------
    frequent_itemsets : pandas.DataFrame
        Output :func:`apriori` (wajib memuat kolom ``itemsets`` dan ``support``).
    metric : str, default "confidence"
        Metrik penyaring: salah satu dari support, confidence, lift,
        leverage, conviction, zhangs_metric.
    min_threshold : float, default 0.5
        Aturan disimpan jika ``metric >= min_threshold``.

    Returns
    -------
    pandas.DataFrame
        Satu baris per aturan, diurutkan berdasarkan lift lalu confidence.
    """
    if metric not in _RULE_METRICS:
        raise ValueError(f"metric harus salah satu dari {_RULE_METRICS}.")
    missing_cols = {"itemsets", "support"} - set(frequent_itemsets.columns)
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing_cols)}")

    support_of = {frozenset(s): float(v) for s, v in
                  zip(frequent_itemsets["itemsets"], frequent_itemsets["support"])}

    rows = []
    for itemset, s_ac in support_of.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset, key=str)
        for r in range(1, len(items)):
            for antecedent_items in combinations(items, r):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                try:
                    s_a = support_of[antecedent]
                    s_c = support_of[consequent]
                except KeyError as exc:
                    raise KeyError(
                        "Subset itemset tidak ditemukan. Pastikan input berasal dari apriori() "
                        "tanpa penyaringan tambahan (sifat downward closure harus terpenuhi)."
                    ) from exc

                confidence = s_ac / s_a
                lift = confidence / s_c
                leverage = s_ac - s_a * s_c
                conviction = np.inf if confidence >= 1 - _EPS else (1 - s_c) / (1 - confidence)
                denom = max(s_ac * (1 - s_a), s_a * (s_c - s_ac))
                # Dibatasi ke [-1, 1] untuk meredam galat pembulatan floating point.
                zhang = float(np.clip(leverage / denom, -1.0, 1.0)) if denom > _EPS else 0.0

                rows.append({
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "antecedent_support": s_a,
                    "consequent_support": s_c,
                    "support": s_ac,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                    "zhangs_metric": zhang,
                })

    columns = ["antecedents", "consequents", "antecedent_support", "consequent_support",
               *_RULE_METRICS]
    rules = pd.DataFrame(rows, columns=columns)
    if rules.empty:
        return rules
    rules = rules[rules[metric] >= min_threshold - _EPS]
    return rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def format_itemset(itemset: Iterable) -> str:
    """Representasi string yang rapi, contoh: ``{kopi, gula}``."""
    return "{" + ", ".join(sorted(map(str, itemset))) + "}"
