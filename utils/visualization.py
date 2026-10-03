from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_dataset(X: np.ndarray, y: np.ndarray, title: str, labels: dict[str, str]):
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="white", s=55)
    ax.legend(*scatter.legend_elements(), title=labels["class_label"], loc="best")
    ax.set(title=title, xlabel=labels["feature_1"], ylabel=labels["feature_2"])
    fig.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_decision_boundary(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str,
    labels: dict[str, str],
    resolution: int = 80,
):
    padding = 0.25
    x_min, x_max = X_train[:, 0].min() - padding, X_train[:, 0].max() + padding
    y_min, y_max = X_train[:, 1].min() - padding, X_train[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = predict_fn(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="white", label=labels["train"])
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolor="black", marker="X", s=72, label=labels["test"])
    ax.set(title=title, xlabel=labels["feature_1"], ylabel=labels["feature_2"])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_objective(values: list[float], labels: dict[str, str]):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3)
    ax.set(xlabel=labels["evaluation"], ylabel=labels["loss"], title=labels["objective"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
