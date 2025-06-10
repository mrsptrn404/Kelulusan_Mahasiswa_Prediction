import streamlit as st
from pages import EDA, Modeling, Predict

st.set_page_config(page_title="Prediksi Kelulusan", layout="wide")

# Halaman utama
st.title("🎓 Dashboard Prediksi Kelulusan Mahasiswa")
st.markdown("""
## 👋 Selamat Datang di Dashboard Prediksi Kelulusan Mahasiswa!

Aplikasi ini digunakan untuk memprediksi **kelulusan cepat mahasiswa** berdasarkan:
- IPK
- Jumlah pelatihan pengembangan diri
- Jumlah prestasi
- Keterlibatan dalam forum komunikasi selama kuliah
- Jumlah kegiatan organisasi

🧠 Didukung oleh model Machine Learning (Naive Bayes & KNN).

Dashboard ini terdiri dari tiga bagian utama:
1. Exploratory Data Analysis (EDA): Proses awal yang bertujuan untuk memahami data secara menyeluruh sebelum membangun model prediksi. Melalui EDA, kita dapat melihat struktur data, mengenali pola-pola penting, hubungan antar variabel, serta mendeteksi anomali atau data yang tidak wajar. Proses ini membantu memastikan bahwa data yang digunakan bersih, relevan, dan siap untuk dianalisis lebih lanjut. Selain itu, EDA juga memberikan wawasan awal mengenai seberapa besar pengaruh fitur-fitur seperti IPK, prestasi, dan keterlibatan organisasi terhadap kelulusan cepat, sehingga dapat meningkatkan akurasi dan interpretabilitas model yang dibangun.
2. Modeling: Proses membangun dan melatih model machine learning berdasarkan data yang telah dianalisis sebelumnya, dengan tujuan untuk mempelajari pola dan hubungan antar variabel sehingga dapat digunakan untuk melakukan prediksi. Dalam konteks prediksi kelulusan mahasiswa, tahap modeling digunakan untuk melatih algoritma seperti Naive Bayes dan K-Nearest Neighbors (KNN) agar dapat memprediksi apakah seorang mahasiswa akan lulus tepat waktu berdasarkan data seperti IPK, jumlah pelatihan, prestasi, keaktifan dalam forum komunikasi selama kuliah, dan keterlibatan organisasi. Hasil dari modeling ini berupa model terlatih yang kemudian dapat digunakan untuk menguji akurasi dan melakukan prediksi pada data baru.
3. Prediction: Tahap penggunaan model yang telah dilatih sebelumnya untuk memprediksi atau memperkirakan hasil dari data baru yang belum diketahui hasilnya. Dalam konteks aplikasi ini, prediction digunakan untuk memproyeksikan apakah seorang mahasiswa akan lulus cepat atau tidak berdasarkan data yang dimasukkan, seperti IPK, jumlah pelatihan, prestasi, keterlibatan dalam forum komunikasi, dan kegiatan organisasi. Tujuan dari tahap ini adalah memberikan gambaran atau rekomendasi berbasis data yang dapat membantu mahasiswa, dosen, atau pihak kampus dalam mengambil keputusan atau memberikan dukungan akademik secara lebih tepat sasaran.

📌 **Silakan pilih halaman yang ingin Anda tuju melalui navigasi sidebar di sebelah kiri.**  
➡️ **Gunakan menu navigasi di kiri layar untuk menjelajahi fitur-fitur dashboard.**
""")
