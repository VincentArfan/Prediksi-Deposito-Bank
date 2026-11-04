# Klasifikasi Keberhasilan Pemasaran Bank Menggunakan MLP

## Deskripsi Project

Project ini bertujuan membangun model Machine Learning berbasis **Multi-Layer Perceptron (MLP)** untuk memprediksi keberhasilan kampanye pemasaran bank dalam memperoleh nasabah yang berlangganan deposito berjangka (_term deposit_).

Model dikembangkan menggunakan dataset **Bank Marketing** dari UCI Machine Learning Repository. Pengembangan dilakukan melalui beberapa tahapan, mulai dari eksplorasi data, preprocessing, feature engineering, pelatihan model, optimasi, hingga evaluasi performa.

Project ini mencakup pengembangan model untuk Tugas 1 dan Ujian Tengah Semester (UTS), dengan eksperimen dan pengembangan model yang didokumentasikan secara terpisah.

## Tujuan Project

1. Membangun model klasifikasi biner menggunakan algoritma Multi-Layer Perceptron (MLP).
2. Menganalisis karakteristik dataset Bank Marketing.
3. Melakukan preprocessing dan feature engineering untuk meningkatkan kualitas fitur.
4. Menangani ketidakseimbangan kelas menggunakan Balanced Class Weighting.
5. Meningkatkan keputusan klasifikasi melalui Threshold Tuning.
6. Membandingkan performa model MLP Baseline dan MLP Tuned.
7. Mengevaluasi kontribusi fitur menggunakan Permutation Importance dan Ablation Study.

## Dataset

Dataset yang digunakan adalah **Bank Marketing Dataset** dari UCI Machine Learning Repository.

- Sumber: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Jumlah data: 45.211 baris.
- Target: `y`
- Kelas 0: `No` (tidak berlangganan deposito).
- Kelas 1: `Yes` (berlangganan deposito).

## Feature Engineering

Feature engineering dilakukan untuk menghasilkan informasi tambahan dari fitur yang tersedia pada dataset.

Pada pengembangan model Tugas 1, digunakan dua fitur tambahan:

1. **`pdays_contacted`**: fitur biner yang menunjukkan apakah nasabah pernah dihubungi pada kampanye sebelumnya.
2. **`balance_per_age`**: fitur numerik yang merupakan rasio saldo rekening terhadap usia nasabah.

Pada pengembangan model UTS, dilakukan pengembangan lebih lanjut dengan menggunakan empat fitur tambahan:

1. **`pdays_contacted`**: fitur biner yang menunjukkan apakah nasabah pernah dihubungi pada kampanye sebelumnya.
2. **`balance_per_age`**: fitur numerik yang merupakan rasio saldo rekening terhadap usia nasabah.
3. **`pdays_recency_group`**: fitur kategorikal yang mengelompokkan nasabah berdasarkan rentang waktu sejak terakhir kali dihubungi pada kampanye sebelumnya.
4. **`campaign_intensity`**: fitur kategorikal yang mengelompokkan nasabah berdasarkan jumlah kontak dalam kampanye pemasaran yang sedang berjalan.

## Metodologi

Tahapan pengembangan model meliputi:

1. Data Loading dan Exploratory Data Analysis (EDA).
2. Data Cleaning dan Preprocessing.
3. Feature Engineering.
4. Encoding fitur kategorikal.
5. Standardisasi fitur numerik.
6. Pembagian data menjadi training, validation, dan testing.
7. Pelatihan model MLP Baseline.
8. Pengembangan dan optimasi model MLP Tuned.
9. Balanced Class Weighting untuk menangani ketidakseimbangan kelas.
10. Threshold Tuning untuk menentukan ambang klasifikasi.
11. Evaluasi menggunakan metrik klasifikasi.
12. Permutation Importance dan Ablation Study.

## Struktur Repository

```text
Prediksi-Deposito-Bank/
│
├── App/
│   ├── app_tugas1.py
│   └── app_uts.py
│
├── Laporan/
│   ├── DeepLearning_Tugas1.docx
│   ├── DeepLearning_Tugas1.pdf
│   ├── DeepLearning_UTS.docx
│   └── DeepLearning_UTS.pdf
│
├── Tugas_1/
│   ├── artifacts/
│   └── DeepLearning_Tugas1.ipynb
│
├── UTS/
│   ├── artifacts/
│   └── DeepLearning_UTS.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
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
cd Prediksi-Deposito-Bank
```

### 3. Buat Virtual Environment (Opsional)

```bash
python -m venv venv
```

Aktifkan virtual environment:

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi Streamlit

**Aplikasi Tugas 1:**

```bash
streamlit run App/app_tugas1.py
```

**Aplikasi UTS:**

```bash
streamlit run App/app_uts.py
```

Pastikan seluruh file model dan artefak preprocessing yang dibutuhkan tersedia di folder `Tugas_1/artifacts/` atau `UTS/artifacts/` sesuai dengan aplikasi yang dijalankan.

## Hasil dan Evaluasi

Evaluasi dilakukan dengan membandingkan model MLP Baseline dan MLP Tuned menggunakan beberapa metrik klasifikasi, yaitu:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Analisis tambahan dilakukan menggunakan Permutation Importance untuk mengetahui kontribusi relatif fitur terhadap performa model serta Ablation Study untuk mengevaluasi pengaruh fitur hasil feature engineering.

Hasil akhir dan pembahasan eksperimen dapat dilihat pada notebook dan laporan project yang tersedia di repository ini.

## Dokumentasi

Dokumentasi lengkap mengenai metodologi, proses eksperimen, hasil evaluasi, dan analisis model tersedia dalam notebook dan laporan berikut:

- **Tugas 1:** `Tugas_1/DeepLearning_Tugas1.ipynb`
- **Laporan Tugas 1:** `Laporan/DeepLearning_Tugas1.pdf`
- **UTS:** `UTS/DeepLearning_UTS.ipynb`
- **Laporan UTS:** `Laporan/DeepLearning_UTS.pdf`

## Identitas Kelompok

- **Kelompok:** Capcay
- **Kelas:** IF-C Sore
- **Universitas:** Universitas Mikroskil

## Anggota Kelompok

| No. | NIM       | Nama          |
| --- | --------- | ------------- |
| 1   | 231110273 | Febri Sujaya  |
| 2   | 231110613 | Andre Derizhy |
| 3   | 231111378 | Fandy Sanusi  |
| 4   | 231110663 | Vincent Arfan |
| 5   | 231112986 | Bryan Lyanto  |
