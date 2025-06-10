import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🎯 Formulir Prediksi Kelulusan Mahasiswa")

st.markdown("""
Halaman ini bertujuan untuk **melakukan prediksi kelulusan cepat mahasiswa** berdasarkan data input pengguna.  
Sistem menggunakan dua model machine learning:
- **Naive Bayes** (berbasis probabilitas)
- **K-Nearest Neighbors (KNN)** (berbasis kemiripan)

Model ini juga menampilkan **akurasi**, **confusion matrix**, dan **classification report** sebagai evaluasi performa.
""")

# Load dan persiapan data
df = pd.read_csv("lulus.csv")
df.columns = df.columns.str.strip().str.lower()

fitur = ['ipk', 'pelatihan pengembangan diri', 'prestasi', 'forum komunikasi kuliah', 'kegiatan organisasi']
target = 'lulus cepat'

fitur_ada = [col for col in fitur if col in df.columns]
if target not in df.columns:
    st.error("❌ Kolom target 'lulus cepat' tidak ditemukan.")
    st.stop()

X = df[fitur_ada]
y = df[target]

# ===============================
# Training Model
# ===============================
model_nb = GaussianNB()
model_nb.fit(X, y)
y_pred_nb = model_nb.predict(X)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X, y)
y_pred_knn = model_knn.predict(X)

# ===============================
# Evaluasi Model
# ===============================
st.subheader("📈 Evaluasi Model")

# Akurasi
acc_nb = accuracy_score(y, y_pred_nb)
acc_knn = accuracy_score(y, y_pred_knn)
st.write(f"**Akurasi Naive Bayes:** {acc_nb:.2f}")
st.write(f"**Akurasi KNN (k=3):** {acc_knn:.2f}")

# Classification Report
st.write("**Classification Report - Naive Bayes**")
st.dataframe(pd.DataFrame(classification_report(y, y_pred_nb, output_dict=True)).transpose())

st.write("**Classification Report - KNN**")
st.dataframe(pd.DataFrame(classification_report(y, y_pred_knn, output_dict=True)).transpose())

# Confusion Matrix Heatmap
st.write("**Confusion Matrix (Naive Bayes)**")
cm_nb = confusion_matrix(y, y_pred_nb)
fig1, ax1 = plt.subplots()
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

st.write("**Confusion Matrix (KNN)**")
cm_knn = confusion_matrix(y, y_pred_knn)
fig2, ax2 = plt.subplots()
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# ===============================
# Formulir Input Prediksi Baru
# ===============================
st.subheader("📝 Masukkan Data Mahasiswa Baru untuk Prediksi")

ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
pelatihan = st.number_input("Jumlah pelatihan pengembangan diri (0-10)", 0, 10)
prestasi = st.number_input("Jumlah prestasi (0-50)", 0, 50)
komunikasi = st.number_input("Keterlibatan dalam forum komunikasi (0-10)", 0.0, 10.0, step=0.1)
organisasi = st.number_input("Jumlah kegiatan organisasi (0-10)", 0, 10)

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

    label_map = {1: "Lulus Cepat", 0: "Tidak Lulus Cepat"}
    st.success(f"📘 Prediksi Naive Bayes: **{label_map[pred_nb]}**")
    st.success(f"🤖 Prediksi KNN (k=3): **{label_map[pred_knn]}**")
