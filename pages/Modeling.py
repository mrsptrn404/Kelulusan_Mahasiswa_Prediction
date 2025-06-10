import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run():
    st.subheader("🤖 Pelatihan Model")

    # Baca data
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip().str.lower()

    st.write("Kolom yang tersedia di data:")
    st.write(df.columns.tolist())

    # Fitur yang digunakan
    fitur_input = ['ipk', 'pelatihan pengembangan diri', 'prestasi', 'kegiatan organisasi', 'forum komunikasi kuliah']
    fitur_ada = [col for col in fitur_input if col in df.columns]

    if 'lulus cepat' not in df.columns:
        st.error("Kolom target 'lulus cepat' tidak ditemukan!")
        return

    if len(fitur_ada) < len(fitur_input):
        st.warning(f"Kolom berikut tidak ditemukan dan tidak digunakan: {set(fitur_input) - set(fitur_ada)}")

    # Persiapan data
    X = df[fitur_ada]
    y = df['lulus cepat']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # =====================
    # Naive Bayes
    # =====================
    st.markdown("### 📘 Naive Bayes Classifier")

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb, output_dict=True)

    st.success(f"Akurasi Naive Bayes: {acc_nb:.2f}")
    st.write("**Confusion Matrix Naive Bayes**")
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

    st.write("**Classification Report Naive Bayes**")
    st.dataframe(pd.DataFrame(report_nb).transpose())

    # =====================
    # K-Nearest Neighbors
    # =====================
    st.markdown("### 🤖 K-Nearest Neighbors (KNN, k=3)")

    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

    st.success(f"Akurasi KNN (k=3): {acc_knn:.2f}")
    st.write("**Confusion Matrix KNN**")
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

    st.write("**Classification Report KNN**")
    st.dataframe(pd.DataFrame(report_knn).transpose())
