from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from dmtoolkit.association import (
    apriori,
    association_rules,
    encode_transactions,
    format_itemset,
    itemset_support,
)
from dmtoolkit.datasets import make_market_basket

TOY = [
    ["roti", "susu"],
    ["roti", "popok", "bir", "telur"],
    ["susu", "popok", "bir", "cola"],
    ["roti", "susu", "popok", "bir"],
    ["roti", "susu", "popok", "cola"],
]


def test_encode_transactions_shape_and_dedup():
    basket = encode_transactions([["a", "b", "a"], ["b"], []])
    assert list(basket.columns) == ["a", "b"]
    assert basket.dtypes.eq(bool).all()
    assert basket.to_numpy().tolist() == [[True, True], [False, True], [False, False]]


def test_apriori_toy_known_supports():
    basket = encode_transactions(TOY)
    freq = apriori(basket, min_support=0.6)
    supports = {fs: s for fs, s in zip(freq["itemsets"], freq["support"])}
    assert supports[frozenset({"roti"})] == pytest.approx(0.8)
    assert supports[frozenset({"popok", "bir"})] == pytest.approx(0.6)
    assert supports[frozenset({"roti", "susu"})] == pytest.approx(0.6)
    # {roti, susu, popok} hanya 2/5 = 0.4 sehingga tidak frequent.
    assert frozenset({"roti", "susu", "popok"}) not in supports


def _brute_force(basket, min_support, max_len):
    out = {}
    for k in range(1, max_len + 1):
        for combo in combinations(basket.columns, k):
            s = itemset_support(basket, combo)
            if s >= min_support - 1e-12:
                out[frozenset(combo)] = s
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_apriori_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    basket = pd.DataFrame(rng.random((120, 8)) < 0.35, columns=[f"i{j}" for j in range(8)])
    expected = _brute_force(basket, 0.08, 4)
    got = apriori(basket, min_support=0.08, max_len=4)
    got_map = {fs: s for fs, s in zip(got["itemsets"], got["support"])}
    assert got_map.keys() == expected.keys()
    for key, value in expected.items():
        assert got_map[key] == pytest.approx(value)


def test_apriori_validation_and_empty():
    basket = encode_transactions(TOY)
    with pytest.raises(ValueError):
        apriori(basket, min_support=0)
    with pytest.raises(ValueError):
        apriori(pd.DataFrame({"a": [0, 2]}), min_support=0.5)
    assert apriori(basket, min_support=1.0).empty


def test_association_rules_metrics():
    basket = encode_transactions(TOY)
    rules = association_rules(apriori(basket, min_support=0.4), metric="confidence", min_threshold=0.0)
    rule = rules[(rules["antecedents"] == frozenset({"bir"})) &
                 (rules["consequents"] == frozenset({"popok"}))].iloc[0]
    # support(bir)=0.6, support(popok)=0.8, support(bir,popok)=0.6
    assert rule["confidence"] == pytest.approx(1.0)
    assert rule["lift"] == pytest.approx(1.25)
    assert rule["leverage"] == pytest.approx(0.6 - 0.6 * 0.8)
    assert np.isinf(rule["conviction"])
    assert -1 <= rules["zhangs_metric"].min() and rules["zhangs_metric"].max() <= 1


def test_association_rules_threshold_filter_and_sorting():
    basket = encode_transactions(TOY)
    rules = association_rules(apriori(basket, min_support=0.4), metric="lift", min_threshold=1.0)
    assert (rules["lift"] >= 1.0 - 1e-12).all()
    assert rules["lift"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        association_rules(apriori(basket, min_support=0.4), metric="unknown")


def test_planted_rules_are_recovered():
    basket = encode_transactions(make_market_basket(3000, random_state=7))
    rules = association_rules(apriori(basket, min_support=0.02), metric="lift", min_threshold=1.5)
    pairs = {(format_itemset(a), format_itemset(c)) for a, c in zip(rules["antecedents"], rules["consequents"])}
    assert ("{roti}", "{selai}") in pairs
    assert ("{popok}", "{susu bayi}") in pairs
