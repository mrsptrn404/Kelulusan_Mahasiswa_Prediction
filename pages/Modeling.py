import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run():
    st.subheader("🤖 Pelatihan Model")

    # Membaca data
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip()  # Menghilangkan spasi tak terlihat

    # Tampilkan kolom untuk debug (opsional)
    st.write("Kolom tersedia dalam data:", df.columns.tolist())

    # Memilih fitur dan target
    X = df[['IPK', 'Pelatihan Pengembangan Diri', 'Prestasi', 'Kegiatan Organisasi']]
    y = df['Lulus Cepat']

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

    # Tampilkan hasil
    st.write("### Akurasi Model")
    st.success(f"Naive Bayes: {acc_nb:.2f}")
    st.success(f"KNN (k=3): {acc_knn:.2f}")
