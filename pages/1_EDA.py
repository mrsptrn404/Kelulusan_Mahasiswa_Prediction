import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Judul dan penjelasan
st.subheader("📊 Exploratory Data Analysis (EDA)")

st.markdown("""
Exploratory Data Analysis (EDA) adalah tahap awal dalam proses analisis data yang bertujuan untuk memahami struktur, karakteristik, dan pola dalam dataset sebelum dilakukan pemodelan.  
Pada halaman ini, Anda dapat melihat:
- Seluruh data mentah
- Statistik deskriptif
- Distribusi variabel target (`Lulus Cepat`)
- Korelasi antar fitur numerik

Dengan EDA, kita dapat menemukan pola penting, mendeteksi data tidak wajar, dan menentukan fitur-fitur yang paling berpengaruh terhadap kelulusan cepat mahasiswa.
""")

# Load data
df = pd.read_csv("lulus.csv")
df.columns = df.columns.str.strip()  # Bersihkan nama kolom dari spasi

# Tampilkan seluruh data
st.write("### 🔍 Seluruh Data")
st.dataframe(df)  # <-- menampilkan semua data, bukan hanya df.head()

# Statistik deskriptif
st.write("### 📈 Statistik Deskriptif")
st.dataframe(df.describe())

# Distribusi target
if 'Lulus Cepat' in df.columns:
    st.write("### 📊 Distribusi Target: Lulus Cepat")
    st.bar_chart(df['Lulus Cepat'].value_counts())
else:
    st.warning("Kolom 'Lulus Cepat' tidak ditemukan dalam data!")

# Korelasi antar variabel numerik
st.write("### 🔗 Korelasi Antar Variabel Numerik")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 6))  # Lebih besar agar lebih jelas
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
