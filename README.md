# Panduan Belajar Data Mining

Repositori ini berisi materi pembelajaran data mining praktis berbasis Python dan scikit-learn. Materi disusun secara bertahap dari pemahaman data dasar hingga implementasi pipeline machine learning end-to-end, lalu ditutup dengan teknik tingkat lanjut yang siap produksi.

## Struktur Modul

- `00_indeks_belajar_data_mining.ipynb`: Peta jalan belajar dan pemeriksaan dependensi.
- `01_level_light.ipynb`: Exploratory Data Analysis (EDA), statistik deskriptif, visualisasi, dan deteksi anomali data.
- `02_level_medium.ipynb`: Klasifikasi dasar (KNN, Decision Tree, Logistic Regression), pencegahan data leakage via Pipeline, dan cross-validation.
- `03_level_high.ipynb`: Hyperparameter tuning (GridSearchCV), ensemble Random Forest, evaluasi dataset imbalanced, dan permutation importance.
- `04_level_extra_high.ipynb`: Proyek tabular end-to-end dengan ColumnTransformer, optimasi threshold prediksi, dan clustering K-Means.
- `05_level_expert.ipynb`: Data mining tingkat lanjut dan siap produksi (lihat rincian di bawah).
- `belajar_data_mining_bertahap.ipynb`: Ringkasan seluruh level dalam satu notebook.

### Level 5: Expert

| Bagian | Topik | Teknik |
|:---|:---|:---|
| A | Association Rule Mining | Apriori (implementasi sendiri), support, confidence, lift, leverage, conviction |
| B | Anomaly Detection | Robust Z-Score (MAD), Isolation Forest, Local Outlier Factor, masking effect |
| C | Model Tabular Lanjutan | HistGradientBoosting (native categorical & missing), Stacking, perbandingan berpasangan per fold, Nested CV |
| D | Kalibrasi Probabilitas | Brier score, Expected Calibration Error, reliability diagram, sigmoid/isotonic |
| E | Keputusan Berbasis Biaya | Matriks biaya, threshold Bayes-optimal, kurva biaya, lift/gain chart |
| F | Evaluasi Akhir | Bootstrap confidence interval untuk metrik dan penghematan biaya |
| G | Interpretasi Model | Permutation importance, Partial Dependence, ICE |
| H | Reduksi Dimensi | PCA, scree plot, PCA di dalam pipeline |
| I | Deployment & Monitoring | joblib, Population Stability Index (PSI), experiment tracking, model card |

Seluruh dataset Level 5 **disimulasikan** dengan pola yang sengaja ditanam, sehingga hasil setiap algoritma dapat diverifikasi terhadap "kebenaran" yang diketahui.

## Paket Pendukung `dmtoolkit`

Fungsi-fungsi yang dipakai Level 5 dikemas sebagai paket Python kecil yang dapat digunakan ulang pada proyek lain:

| Modul | Isi |
|:---|:---|
| `dmtoolkit.association` | `encode_transactions`, `apriori`, `association_rules`, `itemset_support` |
| `dmtoolkit.evaluation` | `bayes_optimal_threshold`, `optimal_threshold`, `threshold_cost_curve`, `expected_calibration_error`, `reliability_table`, `bootstrap_ci`, `lift_table`, `nested_cross_validation` |
| `dmtoolkit.drift` | `population_stability_index`, `drift_report` (PSI + uji Kolmogorov-Smirnov) |
| `dmtoolkit.datasets` | `make_market_basket`, `make_churn_dataset` (dengan opsi drift), `make_transaction_anomalies` |
| `dmtoolkit.experiment` | `ExperimentTracker` (JSON Lines), `dataframe_fingerprint`, `ModelCard` |

Contoh singkat:

```python
from dmtoolkit import apriori, association_rules, encode_transactions
from dmtoolkit.datasets import make_market_basket

basket = encode_transactions(make_market_basket(3000))
rules = association_rules(apriori(basket, min_support=0.02), metric="lift", min_threshold=1.5)
print(rules.head())
```

## Kebutuhan Library

- Python 3.9+
- numpy, pandas, matplotlib, seaborn
- scikit-learn, scipy, joblib

Instalasi:

```bash
pip install -r requirements.txt
```

Untuk pengembangan (unit test dan eksekusi notebook otomatis):

```bash
pip install -r requirements-dev.txt
pip install -e .            # opsional: memasang dmtoolkit sebagai paket
```

## Cara Menjalankan

1. Buka terminal di direktori proyek.
2. Jalankan server Jupyter:

```bash
jupyter notebook
```

3. Buka notebook secara berurutan mulai dari level 1 (`01_level_light.ipynb`) hingga level 5 (`05_level_expert.ipynb`).

Output Level 5 (model `joblib`, log eksperimen, model card) disimpan di folder `artifacts/` yang diabaikan oleh git.

## Pengujian

```bash
python -m pytest                                        # unit test dmtoolkit
python scripts/run_notebooks.py                         # eksekusi semua notebook secara headless
python scripts/run_notebooks.py 05_level_expert.ipynb   # atau notebook tertentu saja
```

Workflow GitHub Actions (`.github/workflows/ci.yml`) menjalankan kedua pemeriksaan tersebut pada setiap push ke `main` dan setiap pull request.

## Struktur Direktori

```
.
├── 00_indeks_belajar_data_mining.ipynb
├── 01_level_light.ipynb ... 05_level_expert.ipynb
├── belajar_data_mining_bertahap.ipynb
├── dmtoolkit/              # paket utilitas Level 5
├── tests/                  # unit test (pytest)
├── scripts/run_notebooks.py
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```
