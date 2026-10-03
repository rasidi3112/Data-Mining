from time import perf_counter

import numpy as np
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.primitives import QMLSampler
from qiskit_machine_learning.utils import algorithm_globals


def build_circuit(num_qubits: int = 2):
    feature_map = zz_feature_map(num_qubits, reps=1)
    ansatz = real_amplitudes(num_qubits, reps=1)
    return feature_map, ansatz, feature_map.compose(ansatz)


def train_quantum_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    maxiter: int = 25,
    seed: int = 42,
) -> tuple[VQC, float, list[float]]:
    algorithm_globals.random_seed = seed
    feature_map, ansatz, _ = build_circuit(num_qubits=X_train.shape[1])
    objective_values: list[float] = []

    def record_objective(_weights, value):
        objective_values.append(float(value))

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        loss="cross_entropy",
        sampler=QMLSampler(shots=None),
        callback=record_objective,
    )
    start = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - start, objective_values


def predict(model: VQC, X: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(X))
    if values.ndim == 2:
        return np.argmax(values, axis=1)
    return values.astype(int).ravel()


def circuit_as_text() -> str:
    _, _, circuit = build_circuit()
    return str(circuit.draw(output="text", fold=100))
