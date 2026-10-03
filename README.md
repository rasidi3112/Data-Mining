# Quantum AI Mini Lab

A small project for seeing the difference between a **classical AI** pipeline and **Quantum Machine Learning (QML)** firsthand. The application creates a two-class dataset, runs Logistic Regression as a baseline, and then trains a two-qubit **Variational Quantum Classifier (VQC)** using Qiskit's local statevector simulator.

> The goal is to learn and experiment, not to prove *quantum advantage*. With small datasets and a simulator, the classical model will usually be faster. That is a valid and useful result to present.

## What you will learn

| Component | Role in the lab |
| --- | --- |
| `make_moons` / `make_blobs` dataset | A two-class classification problem with two features. |
| Logistic Regression | A fast, easy-to-explain classical AI baseline. |
| Qubit | The two input features are mapped to rotation angles on two qubits. |
| `ZZFeatureMap` | A feature map that encodes the data and creates correlations between qubits. |
| `RealAmplitudes` | A parameterized circuit (ansatz) whose weights are trained. |
| VQC + COBYLA | A hybrid loop where a classical optimizer updates quantum circuit parameters. |
| QMLSampler | A local statevector simulator; no account or quantum hardware is required. |

## Project structure

```text
quantum-ai-mini-lab/
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                      # This guide
├── data/                          # Optional location for experiment data
├── models/
│   ├── classical_model.py          # Logistic Regression baseline
│   └── quantum_model.py            # Two-qubit VQC on a simulator
├── utils/
│   └── visualization.py            # Dataset, boundary, loss, and confusion-matrix charts
└── notebooks/
    └── experiment.ipynb            # Step-by-step exploration notebook
```

## Install and run

Prerequisite: Python **3.10+** is recommended and `pip` must be available in your terminal.

```bash
cd quantum-ai-mini-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Streamlit will show a local URL, usually `http://localhost:8501`. Select a dataset, then press **Run Both**. Start with 40–60 samples and 15–25 iterations to keep the experiment quick.

To use the notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```

## Expected output

Exact results depend on your OS, CPU, library versions, and experiment settings. With `make_moons`, 60 samples, seed 42, and 25 iterations, the result table should resemble:

```text
Model                           Accuracy    Training time (s)
Classical: Logistic Regression  ~80–95%     ~0.00–0.10
Quantum: 2-qubit VQC            ~55–90%     ~a few seconds or more
```

The dashboard also shows:

- a text diagram of the quantum circuit;
- a confusion matrix for each model;
- a decision boundary, using a deliberately small quantum grid to keep the simulator responsive;
- the VQC objective-value curve during optimization.

## How the quantum circuit works

1. Two feature values are normalized to the `0..π` range. These values can then be used as rotation-gate angles.
2. `ZZFeatureMap` encodes the input values into qubits. Its `ZZ` component can create two-qubit correlations, providing a simple intuition for correlation and entanglement in a feature map.
3. `RealAmplitudes` adds parameterized rotation gates and entanglement layers. Its parameters are analogous to learnable weights.
4. QMLSampler simulates circuit measurements with a deterministic statevector (`shots=None`). No task is submitted to real quantum hardware.
5. COBYLA, a classical optimizer, changes the ansatz parameters to reduce cross-entropy loss. This is a **hybrid quantum-classical** algorithm.
6. The VQC reads the measurement distribution and converts it into a class-0 or class-1 prediction.

In short:

```text
[two features] → [ZZ feature map] → [trainable ansatz] → [measure / simulator]
                                            ↑                    ↓
                                   COBYLA optimizer ← loss / label
```

## Classical AI vs Quantum AI

| Aspect | Logistic Regression | VQC in this project |
| --- | --- | --- |
| Representation | Linear weights over numeric features | Two-qubit state and gate parameters |
| Training | Classical optimization solver | Classical COBYLA + quantum circuit evaluations |
| Computation | Standard CPU | Statevector simulator on a standard CPU |
| Time on a small example | Usually very fast | Slower due to repeated circuit evaluations |
| Learning value | Baseline with simple interpretation | Learn encoding, ansatz, measurement, and hybrid optimization |
| Performance claim | Practical reference | Concept demonstration, not quantum advantage |

## Qiskit compatibility notes

The implementation uses the modern Qiskit Machine Learning API: `VQC(..., sampler=QMLSampler(shots=None))`. It does **not** use `QuantumInstance` or `qiskit.opflow`, because those patterns have been retired from current Qiskit workflows. The version ranges in `requirements.txt` keep the project within the compatible Qiskit 2.x and `qiskit-machine-learning` 0.9.x families.

## Next development ideas

1. Add `SVC` or Random Forest as a second classical baseline.
2. Compare other feature maps (`ZFeatureMap`, `PauliFeatureMap`) and ansatz `reps` values.
3. Add cross-validation and a CSV experiment table in `data/`.
4. Display prediction probabilities and uncertainty variation from finite-shot simulations.
5. Test a quantum kernel (`QSVC`) to compare two QML approaches.
6. Run a small circuit on IBM Quantum hardware after learning about noise, transpilation, and shot limits.

## File guide

- `app.py` controls the interface, dataset creation, angle scaling, and result comparison.
- `models/classical_model.py` contains the Logistic Regression baseline and timing logic.
- `models/quantum_model.py` builds the circuit and trains the VQC on a simulator.
- `utils/visualization.py` keeps the Matplotlib chart code separate from the UI.
- `notebooks/experiment.ipynb` provides a linear exploration path before using the dashboard.
