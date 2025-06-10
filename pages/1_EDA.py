import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Judul dan penjelasan
st.subheader("📊 Exploratory Data Analysis (EDA)")

st.markdown("""
Exploratory Data Analysis (EDA) adalah tahap awal dalam proses analisis data yang bertujuan untuk memahami struktur, karakteristik, dan pola dalam dataset sebelum dilakukan pemodelan.  
Pada halaman ini, Anda dapat melihat:
- Contoh data awal
- Statistik deskriptif
- Distribusi variabel target (`Lulus Cepat`)
- Korelasi antar fitur numerik

Dengan EDA, kita dapat menemukan pola penting, mendeteksi data tidak wajar, dan menentukan fitur-fitur yang paling berpengaruh terhadap kelulusan cepat mahasiswa.
""")

# Load data
df = pd.read_csv("lulus.csv")
df.columns = df.columns.str.strip()  # Bersihkan nama kolom dari spasi

# Tampilkan sample data
st.write("### 🔍 Contoh Data")
st.dataframe(df.head())

# Statistik deskriptif
st.write("### 📈 Statistik Deskriptif")
st.dataframe(df.describe())

# Distribusi target
if 'Lulus Cepat' in df.columns:
    st.write("### 📊 Distribusi Target: Lulus Cepat")
    st.bar_chart(df['Lulus Cepat'].value_counts())
else:
    st.warning("Kolom 'Lulus Cepat' tidak ditemukan dalam data!")

# Korelasi
st.write("### 🔗 Korelasi Antar Variabel Numerik")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
