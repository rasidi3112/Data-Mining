from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_classical_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegression, float]:
    model = LogisticRegression(max_iter=1_000, random_state=42)
    start = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - start


def predict(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(X)
