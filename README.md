# 🎓 Prediksi Kelulusan Mahasiswa - Dashboard Streamlit

Aplikasi **interaktif berbasis Streamlit** untuk memprediksi apakah seorang mahasiswa akan **lulus cepat atau tidak**, menggunakan model **Naive Bayes** dan **K-Nearest Neighbors (KNN)** berdasarkan data akademik dan aktivitas mahasiswa.

---

## 🚀 Fitur Aplikasi

🔍 **EDA (Exploratory Data Analysis)**  
Visualisasi dan eksplorasi dataset, termasuk statistik deskriptif dan heatmap korelasi.

🧠 **Modeling (Pelatihan Model)**  
Melatih dua algoritma machine learning:  
- **Naive Bayes**
- **K-Nearest Neighbors (KNN)**  
Ditampilkan pula akurasi masing-masing model.

📊 **Prediksi**  
Formulir input interaktif untuk memasukkan data mahasiswa dan mendapatkan hasil prediksi secara langsung.

---

## 📁 Dataset

Dataset digunakan dari [Kaggle - Dataset Kelulusan Mahasiswa](https://www.kaggle.com/datasets/christopherbayuaji/dataset-kelulusan):

### 📌 Fitur:
- `IPK`: Nilai IPK (skala 0.0 - 4.0)
- `Pelatihan Pengetahuan`: Jumlah pelatihan yang diikuti (0-10)
- `Prestasi`: Jumlah prestasi akademik/non-akademik (0-50)
- `Kegiatan Organisasi`: Jumlah kegiatan organisasi (0-10)

🎯 **Target:**  
- `Lulus Cepat`: `Yes` atau `No`

---

## 🛠️ Instalasi dan Menjalankan

1. **Clone repo** atau upload file ke direktori lokal:
```bash
git clone https://github.com/mrsptrn404/Kelulusan_Mahasiswa_Prediction.git
cd Kelulusan_Mahasiswa_Prediction
