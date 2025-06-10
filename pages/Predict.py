import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def run():
    st.subheader("📊 Prediksi Kelulusan")

    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip().str.lower()

    st.write("Kolom tersedia di data:")
    st.write(df.columns.tolist())

    # Daftar fitur yang diharapkan
    fitur_input = ['ipk', 'pelatihan pengetahuan', 'prestasi', 'kegiatan organisasi']
    fitur_ada = [col for col in fitur_input if col in df.columns]

    if 'lulus cepat' not in df.columns:
        st.error("Kolom target 'lulus cepat' tidak ditemukan!")
        return

    if not fitur_ada:
        st.error("Tidak ada fitur yang ditemukan dari: " + ", ".join(fitur_input))
        return

    X = df[fitur_ada]
    y = df['lulus cepat']

    # Pelatihan dua model
    model_nb = GaussianNB()
    model_nb.fit(X, y)

    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X, y)

    st.write("### Masukkan Data Mahasiswa Baru")
    input_data = {}
    for col in fitur_ada:
        input_data[col] = st.number_input(f"Masukkan nilai {col.title()}", min_value=0.0, step=0.1)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        pred_nb = model_nb.predict(input_df)[0]
        pred_knn = model_knn.predict(input_df)[0]

        st.success(f"Naive Bayes Prediksi: {'Lulus Cepat' if pred_nb == 1 else 'Tidak Lulus Cepat'}")
        st.success(f"KNN (k=3) Prediksi: {'Lulus Cepat' if pred_knn == 1 else 'Tidak Lulus Cepat'}")
