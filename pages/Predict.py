import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def run():
    st.title("🎯 Prediksi Kelulusan Mahasiswa")

    # Load dan siapkan data
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip().str.lower()

    # Pastikan kolom yang dibutuhkan tersedia
    fitur = ['ipk', 'pelatihan pengembangan diri', 'prestasi', 'forum komunikasi kuliah', 'kegiatan organisasi']
    target = 'lulus cepat'

    fitur_ada = [col for col in fitur if col in df.columns]

    if target not in df.columns:
        st.error("Kolom target 'lulus cepat' tidak ditemukan dalam dataset.")
        return
    if len(fitur_ada) < len(fitur):
        st.warning(f"Beberapa fitur tidak ditemukan di data: {set(fitur) - set(fitur_ada)}")
        return

    # Data pelatihan model
    X = df[fitur]
    y = df[target]

    model_nb = GaussianNB()
    model_nb.fit(X, y)

    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X, y)

    # Input data pengguna
    st.subheader("📝 Masukkan Data Mahasiswa")
    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
    pelatihan = st.number_input("Jumlah pelatihan pengembangan diri yang diikuti (0-10)", min_value=0, max_value=10)
    prestasi = st.number_input("Jumlah prestasi yang diperoleh (0-50)", min_value=0, max_value=50)
    komunikasi = st.number_input("Keterlibatan dalam forum komunikasi selama kuliah", min_value=0.0, max_value=10.0, step=0.1)
    organisasi = st.number_input("Jumlah kegiatan organisasi yang diikuti (0-10)", min_value=0, max_value=10)

    # Prediksi
    if st.button("🔮 Prediksi"):
        input_data = pd.DataFrame([{
            'ipk': ipk,
            'pelatihan pengembangan diri': pelatihan,
            'prestasi': prestasi,
            'forum komunikasi kuliah': komunikasi,
            'kegiatan organisasi': organisasi
        }])

        pred_nb = model_nb.predict(input_data)[0]
        pred_knn = model_knn.predict(input_data)[0]

        st.success(f"🎓 Naive Bayes Prediksi: {'Lulus Cepat' if pred_nb == 1 else 'Tidak Lulus Cepat'}")
        st.success(f"🤖 KNN (k=3) Prediksi: {'Lulus Cepat' if pred_knn == 1 else 'Tidak Lulus Cepat'}")
