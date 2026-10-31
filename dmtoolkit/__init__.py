"""dmtoolkit - utilitas data mining untuk materi Level 5 (Expert).

Sub-modul:

- ``association`` : Apriori & association rules (market basket analysis)
- ``evaluation``  : threshold berbasis biaya, kalibrasi, bootstrap CI,
                    lift/gain table, nested cross-validation
- ``drift``       : Population Stability Index & laporan data drift
- ``datasets``    : generator dataset sintetis yang deterministik
- ``experiment``  : pencatatan eksperimen, sidik jari data, model card
"""

from . import association, datasets, drift, evaluation, experiment
from .association import apriori, association_rules, encode_transactions, itemset_support
from .drift import drift_report, population_stability_index
from .evaluation import (
    bayes_optimal_threshold,
    bootstrap_ci,
    expected_calibration_error,
    lift_table,
    nested_cross_validation,
    optimal_threshold,
)
from .experiment import ExperimentTracker, ModelCard, dataframe_fingerprint

__version__ = "0.1.0"

__all__ = [
    "association", "datasets", "drift", "evaluation", "experiment",
    "apriori", "association_rules", "encode_transactions", "itemset_support",
    "drift_report", "population_stability_index",
    "bayes_optimal_threshold", "bootstrap_ci", "expected_calibration_error",
    "lift_table", "nested_cross_validation", "optimal_threshold",
    "ExperimentTracker", "ModelCard", "dataframe_fingerprint",
]
