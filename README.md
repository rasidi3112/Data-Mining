# Panduan Belajar Data Mining

Repositori ini berisi materi pembelajaran data mining praktis berbasis Python dan scikit-learn. Materi disusun secara bertahap dari pemahaman data dasar hingga implementasi pipeline machine learning end-to-end.

## Struktur Modul

- `00_indeks_belajar_data_mining.ipynb`: Peta jalan belajar dan pemeriksaan dependensi.
- `01_level_light.ipynb`: Exploratory Data Analysis (EDA), statistik deskriptif, visualisasi, dan deteksi anomali data.
- `02_level_medium.ipynb`: Klasifikasi dasar (KNN, Decision Tree, Logistic Regression), pencegahan data leakage via Pipeline, dan cross-validation.
- `03_level_high.ipynb`: Hyperparameter tuning (GridSearchCV), ensemble Random Forest, evaluasi dataset imbalanced, dan permutation importance.
- `04_level_extra_high.ipynb`: Proyek tabular end-to-end dengan ColumnTransformer, optimasi threshold prediksi, dan clustering K-Means.

## Kebutuhan Library

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

Instalasi library dapat dilakukan melalui pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Cara Menjalankan

1. Buka terminal di direktori proyek.
2. Jalankan server Jupyter:

```bash
jupyter notebook
```

3. Buka notebook secara berurutan mulai dari level 1 (`01_level_light.ipynb`) hingga level 4 (`04_level_extra_high.ipynb`).
