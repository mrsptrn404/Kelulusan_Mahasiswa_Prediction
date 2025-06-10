# 🎓 Dashboard Prediksi Kelulusan Mahasiswa

Proyek ini bertujuan untuk memprediksi **kelulusan cepat atau tidaknya mahasiswa** berdasarkan beberapa indikator penting. Aplikasi dibangun menggunakan **Python**, **Streamlit**, dan teknik machine learning sederhana seperti **Naive Bayes** dan **KNN**.

## 📊 Fitur Aplikasi

Aplikasi Streamlit ini terdiri dari tiga halaman utama:

1. **EDA (Exploratory Data Analysis)**  
   Menampilkan analisis deskriptif dan visualisasi awal dari dataset.

2. **Modeling**  
   Menyediakan dua algoritma pembelajaran mesin (Naive Bayes dan K-Nearest Neighbors) untuk pelatihan dan evaluasi model.

3. **Predict**  
   Halaman interaktif untuk memprediksi apakah seorang mahasiswa akan lulus cepat atau tidak berdasarkan input pengguna.

## 📁 Dataset

Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/) dan berisi beberapa fitur penting:

| Fitur                         | Deskripsi                                                      |
|------------------------------|-----------------------------------------------------------------|
| `IPK`                        | Indeks Prestasi Kumulatif mahasiswa                             |
| `Pelatihan Pengembangan Diri`| Jumlah pelatihan yang diikuti terkait pengembangan pribadi      |
| `Prestasi`                   | Jumlah prestasi atau penghargaan yang diperoleh                 |
| `Forum Komunikasi Kuliah`    | Keterlibatan mahasiswa dalam forum komunikasi selama kuliah     |
| `Kegiatan Organisasi`        | Jumlah kegiatan organisasi yang diikuti                         |
| `Lulus Cepat`                | Target variabel: 1 (lulus cepat), 0 (tidak lulus cepat)         |

## ⚙️ Teknologi yang Digunakan

- Python 3.13
- Pandas
- Scikit-learn
- Streamlit

## 🚀 Cara Menjalankan Aplikasi

```bash
streamlit run Dashboard.py
