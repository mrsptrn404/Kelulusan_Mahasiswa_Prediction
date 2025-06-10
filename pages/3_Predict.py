import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🎯 Formulir Prediksi Kelulusan Mahasiswa")

st.markdown("""
Halaman ini bertujuan untuk **melakukan prediksi kelulusan cepat mahasiswa** berdasarkan data input pengguna.  
Sistem akan menggunakan dua model machine learning:
- **Naive Bayes** (berbasis probabilitas)
- **K-Nearest Neighbors (KNN)** (berbasis kemiripan data)

Hasil prediksi akan menunjukkan apakah mahasiswa diperkirakan akan **lulus tepat waktu** atau **tidak**,  
disertai metrik evaluasi model seperti akurasi, precision, recall, F1-score, dan confusion matrix.
""")

# Load dan bersihkan data
df = pd.read_csv("lulus.csv")
df.columns = df.columns.str.strip().str.lower()

fitur = ['ipk', 'pelatihan pengembangan diri', 'prestasi', 'forum komunikasi kuliah', 'kegiatan organisasi']
target = 'lulus cepat'

# Validasi kolom
fitur_ada = [col for col in fitur if col in df.columns]
if target not in df.columns:
    st.error("❌ Kolom target 'lulus cepat' tidak ditemukan dalam data.")
    st.stop()
if len(fitur_ada) < len(fitur):
    st.warning(f"⚠️ Kolom berikut tidak ditemukan dan tidak digunakan: {set(fitur) - set(fitur_ada)}")

# Split data
X = df[fitur_ada]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

# Metrik Evaluasi
st.subheader("📈 Evaluasi Model")

# Akurasi
st.write("**Akurasi**")
st.info(f"Naive Bayes: {accuracy_score(y_test, y_pred_nb):.2f}")
st.info(f"KNN (k=3): {accuracy_score(y_test, y_pred_knn):.2f}")

# Confusion Matrix - Naive Bayes
st.write("### Confusion Matrix - Naive Bayes")
cm_nb = confusion_matrix(y_test, y_pred_nb)
fig1, ax1 = plt.subplots()
sns.heatmap(cm_nb, annot=True, fmt='d', cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
st.pyplot(fig1)

# Confusion Matrix - KNN
st.write("### Confusion Matrix - KNN")
cm_knn = confusion_matrix(y_test, y_pred_knn)
fig2, ax2 = plt.subplots()
sns.heatmap(cm_knn, annot=True, fmt='d', cmap="Greens", ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# Classification Report
st.write("### Classification Report - Naive Bayes")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_nb, output_dict=True)).transpose())

st.write("### Classification Report - KNN")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_knn, output_dict=True)).transpose())

# Prediksi baru
st.subheader("📝 Masukkan Data Mahasiswa untuk Prediksi")

ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
pelatihan = st.number_input("Jumlah pelatihan pengembangan diri yang diikuti (0-10)", min_value=0, max_value=10)
prestasi = st.number_input("Jumlah prestasi yang diperoleh (0-50)", min_value=0, max_value=50)
komunikasi = st.number_input("Keterlibatan dalam forum komunikasi selama kuliah (0-10)", min_value=0.0, max_value=10.0, step=0.1)
organisasi = st.number_input("Jumlah kegiatan organisasi yang diikuti (0-10)", min_value=0, max_value=10)

if st.button("🔮 Prediksi Kelulusan"):
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
    st.success(f"📘 Naive Bayes Prediksi: **{label_map[pred_nb]}**")
    st.success(f"🤖 KNN (k=3) Prediksi: **{label_map[pred_knn]}**")
