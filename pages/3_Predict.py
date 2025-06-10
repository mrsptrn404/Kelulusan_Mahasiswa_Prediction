import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def run():
    st.title("🎯 Formulir Prediksi Kelulusan Mahasiswa")

    st.markdown("""
    Halaman ini bertujuan untuk **melakukan prediksi kelulusan cepat mahasiswa** berdasarkan data input pengguna.  
    Dengan memasukkan informasi seperti IPK, jumlah pelatihan, prestasi, keterlibatan dalam forum komunikasi, dan kegiatan organisasi,  
    sistem akan memproses data tersebut menggunakan dua model machine learning:
    - **Naive Bayes** (berbasis probabilitas)
    - **K-Nearest Neighbors (KNN)** (berbasis kemiripan/kemiripan tetangga terdekat)

    Hasil prediksi akan menunjukkan apakah mahasiswa diperkirakan akan **lulus tepat waktu** atau **tidak**.
    """)

    # Load data
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip().str.lower()

    # Definisi fitur dan target
    fitur = ['ipk', 'pelatihan pengembangan diri', 'prestasi', 'forum komunikasi kuliah', 'kegiatan organisasi']
    target = 'lulus cepat'

    fitur_ada = [col for col in fitur if col in df.columns]
    if target not in df.columns:
        st.error("❌ Kolom target 'lulus cepat' tidak ditemukan dalam data.")
        return
    if len(fitur_ada) < len(fitur):
        st.warning(f"⚠️ Kolom berikut tidak ditemukan dan tidak digunakan: {set(fitur) - set(fitur_ada)}")
        return

    # Pelatihan model
    X = df[fitur_ada]
    y = df[target]

    model_nb = GaussianNB()
    model_nb.fit(X, y)

    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X, y)

    # Form input pengguna
    st.subheader("📝 Masukkan Data Mahasiswa")

    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
    pelatihan = st.number_input("Jumlah pelatihan pengembangan diri yang diikuti (0-10)", min_value=0, max_value=10)
    prestasi = st.number_input("Jumlah prestasi yang diperoleh (0-50)", min_value=0, max_value=50)
    komunikasi = st.number_input("Keterlibatan dalam forum komunikasi selama kuliah (0-10)", min_value=0.0, max_value=10.0, step=0.1)
    organisasi = st.number_input("Jumlah kegiatan organisasi yang diikuti (0-10)", min_value=0, max_value=10)

    if st.button("🔮 Prediksi"):
        # Data input baru
        input_data = pd.DataFrame([{
            'ipk': ipk,
            'pelatihan pengembangan diri': pelatihan,
            'prestasi': prestasi,
            'forum komunikasi kuliah': komunikasi,
            'kegiatan organisasi': organisasi
        }])

        # Prediksi
        pred_nb = model_nb.predict(input_data)[0]
        pred_knn = model_knn.predict(input_data)[0]

        # Tampilkan hasil
        label_map = {1: "Lulus Cepat", 0: "Tidak Lulus Cepat"}
        st.success(f"📘 Naive Bayes: **{label_map[pred_nb]}**")
        st.success(f"🤖 KNN (k=3): **{label_map[pred_knn]}**")
