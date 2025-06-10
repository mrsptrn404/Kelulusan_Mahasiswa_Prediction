import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run():
    st.subheader("🤖 Pelatihan Model")

    # Baca data
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip().str.lower()  # Bersihkan dan seragamkan nama kolom

    st.write("Kolom yang tersedia di data:")
    st.write(df.columns.tolist())

    # Daftar fitur yang diinginkan
    fitur_input = ['ipk', 'pelatihan pengetahuan', 'prestasi', 'kegiatan organisasi', 'komunikasi']
    fitur_ada = [col for col in fitur_input if col in df.columns]

    if 'lulus cepat' not in df.columns:
        st.error("Kolom target 'lulus cepat' tidak ditemukan!")
        return

    if not fitur_ada:
        st.error("Tidak ada fitur yang ditemukan dari: " + ', '.join(fitur_input))
        return

    # Siapkan fitur dan target
    X = df[fitur_ada]
    y = df['lulus cepat']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc_nb = accuracy_score(y_test, nb.predict(X_test))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))

    st.write("### Akurasi Model")
    st.success(f"Naive Bayes: {acc_nb:.2f}")
    st.success(f"KNN (k=3): {acc_knn:.2f}")
