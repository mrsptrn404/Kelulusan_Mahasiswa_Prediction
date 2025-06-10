import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run():
    st.subheader("🤖 Pelatihan Model")

    # Load data
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip()

    # Tampilkan kolom untuk pengecekan manual
    st.write("Kolom tersedia dalam dataset:", df.columns.tolist())

    # Fitur dan target
    X = df[['IPK',
            'Pelatihan Pengembangan Diri',
            'Prestasi',
            'Forum Komunikasi Kuliah',
            'Kegiatan Organisasi']]
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

    # Output akurasi
    st.write("### Akurasi Model")
    st.success(f"Naive Bayes: {acc_nb:.2f}")
    st.success(f"KNN (k=3): {acc_knn:.2f}")
