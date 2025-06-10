import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("📊 Eksplorasi Data (EDA)")

df = pd.read_csv("lulus.csv")
df.columns = df.columns.str.strip()

st.write("**Data Sample:**")
st.dataframe(df.head())

st.write("**Statistik Deskriptif:**")
st.dataframe(df.describe())

if 'Lulus Cepat' in df.columns:
    st.write("**Distribusi Kolom Target 'Lulus Cepat':**")
    st.bar_chart(df['Lulus Cepat'].value_counts())
else:
    st.warning("Kolom 'Lulus Cepat' tidak ditemukan!")

st.write("**Korelasi antar variabel numerik:**")
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
