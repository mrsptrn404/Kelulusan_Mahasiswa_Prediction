import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def run():
    st.subheader("📥 Prediksi Kelulusan Mahasiswa")

    # Input
    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
    pelatihan = st.slider("Pelatihan Pengetahuan", 0, 10)
    prestasi = st.slider("Prestasi", 0, 50)
    organisasi = st.slider("Kegiatan Organisasi", 0, 10)

    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip()

    X = df[['IPK', 'Pelatihan Pengetahuan', 'Prestasi', 'Kegiatan Organisasi']]
    y = df['Lulus Cepat']

    # Naive Bayes & KNN
    nb = GaussianNB().fit(X, y)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)

    input_data = pd.DataFrame([[ipk, pelatihan, prestasi, organisasi]],
                              columns=X.columns)

    pred_nb = nb.predict(input_data)[0]
    pred_knn = knn.predict(input_data)[0]

    st.write("### Hasil Prediksi")
    st.info(f"Naive Bayes: **{pred_nb}**")
    st.info(f"KNN: **{pred_knn}**")
