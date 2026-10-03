import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models.classical_model import predict as classical_predict
from models.classical_model import train_classical_model
from models.quantum_model import circuit_as_text, predict as quantum_predict, train_quantum_model
from utils.i18n import LANGUAGE_NAMES, TRANSLATIONS
from utils.visualization import (
    plot_confusion,
    plot_dataset,
    plot_decision_boundary,
    plot_objective,
)


st.set_page_config(page_title="Quantum AI Mini Lab", layout="wide")


@st.cache_data(show_spinner=False)
def make_dataset(name: str, samples: int, noise: float, seed: int):
    if name == "make_moons":
        X, y = make_moons(n_samples=samples, noise=noise, random_state=seed)
    else:
        X, y = make_blobs(
            n_samples=samples, centers=2, n_features=2, cluster_std=1.45, random_state=seed
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test


def result_summary(model_key: str, y_true: np.ndarray, y_pred: np.ndarray, seconds: float) -> dict:
    return {
        "model_key": model_key,
        "accuracy": accuracy_score(y_true, y_pred),
        "time": seconds,
    }


current_language = st.session_state.get("app_language", "Indonesian")
labels = TRANSLATIONS[current_language]

with st.sidebar:
    selected_language = st.selectbox(
        labels["language"],
        list(LANGUAGE_NAMES),
        index=list(LANGUAGE_NAMES).index(current_language),
        format_func=lambda language: LANGUAGE_NAMES[language],
        key="language_selector",
    )

if selected_language != current_language:
    st.session_state["app_language"] = selected_language
    st.rerun()

labels = TRANSLATIONS[selected_language]

st.title(labels["title"])
st.caption(labels["subtitle"])

with st.sidebar:
    st.header(labels["experiment"])
    dataset_name = st.selectbox(labels["dataset"], ["make_moons", "make_blobs"])
    samples = st.slider(labels["samples"], min_value=40, max_value=100, value=60, step=10)
    noise = st.slider(labels["noise"], min_value=0.05, max_value=0.35, value=0.16, step=0.01)
    maxiter = st.slider(labels["iterations"], min_value=5, max_value=50, value=25, step=5)
    seed = st.number_input(labels["seed"], value=42, step=1)

    st.divider()
    st.subheader(labels["concepts"])
    st.markdown("\n".join(f"- {labels[key]}" for key in ("qubit", "superposition", "entanglement", "gate", "vqc")))

X_train, X_test, y_train, y_test = make_dataset(dataset_name, samples, noise, int(seed))
experiment_config = (dataset_name, samples, noise, maxiter, int(seed))

top_left, top_right = st.columns([1, 1])
with top_left:
    st.subheader(labels["data_title"])
    st.pyplot(plot_dataset(np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), dataset_name, labels))
with top_right:
    st.subheader(labels["circuit_title"])
    st.code(circuit_as_text(), language=None)
    st.caption(labels["circuit_caption"])

st.info(labels["info"])

button_left, button_center, button_right = st.columns(3)
run_classical = button_left.button(labels["run_classical"], use_container_width=True)
run_quantum = button_center.button(labels["run_quantum"], type="primary", use_container_width=True)
run_both = button_right.button(labels["run_both"], use_container_width=True)

if run_classical or run_both:
    with st.spinner(labels["spinner_classical"]):
        model, seconds = train_classical_model(X_train, y_train)
        predictions = classical_predict(model, X_test)
    st.session_state["classical"] = {
        "model": model,
        "predictions": predictions,
        "seconds": seconds,
        "summary": result_summary("classical_model", y_test, predictions, seconds),
    }
    st.session_state["result_config"] = experiment_config

if run_quantum or run_both:
    with st.spinner(labels["spinner_quantum"]):
        model, seconds, objectives = train_quantum_model(X_train, y_train, maxiter=maxiter, seed=int(seed))
        predictions = quantum_predict(model, X_test)
    st.session_state["quantum"] = {
        "model": model,
        "predictions": predictions,
        "seconds": seconds,
        "objectives": objectives,
        "summary": result_summary("quantum_model", y_test, predictions, seconds),
    }
    st.session_state["result_config"] = experiment_config

if st.session_state.get("result_config") == experiment_config:
    classical = st.session_state.get("classical")
    quantum = st.session_state.get("quantum")
else:
    classical = quantum = None

if classical or quantum:
    st.divider()
    st.header(labels["results"])
    results = [
        {
            labels["model"]: labels[item["summary"]["model_key"]],
            labels["accuracy"]: item["summary"]["accuracy"],
            labels["time"]: item["summary"]["time"],
        }
        for item in (classical, quantum)
        if item
    ]
    table = pd.DataFrame(results)
    table[labels["accuracy"]] = table[labels["accuracy"]].map("{:.2%}".format)
    table[labels["time"]] = table[labels["time"]].map("{:.3f}".format)
    st.dataframe(table, use_container_width=True, hide_index=True)

    panels = st.columns(2)
    if classical:
        with panels[0]:
            st.subheader(labels["classical_heading"])
            st.pyplot(plot_confusion(y_test, classical["predictions"], labels["classical_cm"]))
            st.pyplot(
                plot_decision_boundary(
                    lambda values: classical_predict(classical["model"], values),
                    X_train, y_train, X_test, y_test, labels["classical_db"], labels
                )
            )
    if quantum:
        with panels[1]:
            st.subheader(labels["quantum_heading"])
            st.pyplot(plot_confusion(y_test, quantum["predictions"], labels["quantum_cm"]))
            st.pyplot(
                plot_decision_boundary(
                    lambda values: quantum_predict(quantum["model"], values),
                    X_train, y_train, X_test, y_test, labels["quantum_db"], labels, resolution=35
                )
            )
            st.pyplot(plot_objective(quantum["objectives"], labels))

    if classical and quantum:
        st.warning(labels["warning"])
else:
    st.caption(labels["empty"])
