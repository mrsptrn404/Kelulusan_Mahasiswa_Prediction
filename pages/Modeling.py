import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def run():
    st.subheader("🤖 Pelatihan Model")

    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip()

    X = df[['IPK', 'Pelatihan Pengetahuan', 'Prestasi', 'Kegiatan Organisasi']]
    y = df['Lulus Cepat']

    # Split
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
