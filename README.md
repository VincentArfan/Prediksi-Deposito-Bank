# Klasifikasi Keberhasilan Pemasaran Bank Menggunakan MLP

## Deskripsi Project

Project ini bertujuan membangun model Machine Learning berbasis **Multi-Layer Perceptron (MLP)** untuk memprediksi keberhasilan kampanye pemasaran bank dalam memperoleh nasabah yang berlangganan deposito berjangka.

Model dikembangkan menggunakan dataset **Bank Marketing** dari UCI Machine Learning Repository. Pengembangan dilakukan melalui tahapan eksplorasi data, preprocessing, feature engineering, pelatihan model, optimasi, dan evaluasi.

## Tujuan Project

1. Membangun model klasifikasi biner menggunakan algoritma Multi-Layer Perceptron (MLP).
2. Menganalisis karakteristik dataset Bank Marketing.
3. Mengembangkan dua fitur tambahan melalui feature engineering.
4. Menangani ketidakseimbangan kelas menggunakan Balanced Class Weighting.
5. Meningkatkan keputusan klasifikasi melalui Threshold Tuning.
6. Membandingkan performa model Baseline dan Tuned.

## Dataset

Dataset yang digunakan adalah **Bank Marketing Dataset** dari UCI Machine Learning Repository.

- Sumber: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Jumlah data: 45.211 baris.
- Target: `y`
- Kelas 0: No (tidak berlangganan deposito).
- Kelas 1: Yes (berlangganan deposito).

## Feature Engineering

Penelitian ini mengembangkan dua fitur tambahan:

1. **`pdays_contacted`**: fitur biner yang menunjukkan apakah nasabah pernah dihubungi pada kampanye sebelumnya.
2. **`balance_per_age`**: fitur numerik yang merupakan rasio saldo rekening terhadap usia nasabah.

## Metodologi

Tahapan pengembangan model meliputi:

1. Data Loading dan Exploratory Data Analysis (EDA).
2. Data Cleaning dan Preprocessing.
3. Feature Engineering.
4. Encoding fitur kategorikal.
5. Standardisasi fitur numerik.
6. Pembagian data menjadi training, validation, dan testing.
7. Pelatihan model MLP Baseline.
8. Pengembangan model MLP Tuned.
9. Balanced Class Weighting untuk menangani ketidakseimbangan kelas.
10. Threshold Tuning untuk menentukan ambang klasifikasi.
11. Evaluasi menggunakan metrik klasifikasi.
12. Permutation Importance dan Ablation Study.

## Struktur Repository

```text
bank-marketing-mlp/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── DeepLearning_Tugas1.ipynb
├── app.py
├── artifacts/
│   ├── bank_marketing_mlp.keras
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_info.json
└── Laporan/
    ├── DeepLearning_Tugas1.docx
    └── DeepLearning_Tugas1.pdf
```

## Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Joblib

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/VincentArfan/Prediksi-Deposito-Bank.git
```

### 2. Masuk ke Folder Project

```bash
cd bank-marketing-mlp
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Pastikan file model dan artefak pendukung tersedia di folder `artifacts/`.

## Hasil dan Evaluasi

Evaluasi dilakukan dengan membandingkan model MLP Baseline dan MLP Tuned menggunakan metrik klasifikasi yang sesuai, seperti:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Hasil akhir dan pembahasan eksperimen dapat dilihat pada notebook dan laporan project.

## Dokumentasi

Penjelasan lengkap mengenai metodologi, eksperimen, hasil evaluasi, dan analisis model tersedia pada notebook dan laporan project di repository ini.

## Anggota Kelompok

| No. | NIM       | Nama          |
| --- | --------- | ------------- |
| 1   | 231110273 | Febri Sujaya  |
| 2   | 231110613 | Andre Derizhy |
| 3   | 231111378 | Fandy Sanusi  |
| 4   | 231110663 | Vincent Arfan |
| 5   | 231112986 | Bryan Lyanto  |
