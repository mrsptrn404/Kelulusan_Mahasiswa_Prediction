import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.subheader("📊 Eksplorasi Data (EDA)")
    
    df = pd.read_csv("lulus.csv")
    df.columns = df.columns.str.strip()  # Bersihkan nama kolom

    st.write("**Data Sample:**")
    st.dataframe(df.head())

    st.write("**Statistik Deskriptif:**")
    st.dataframe(df.describe())

    st.write("**Distribusi Kolom Target 'Lulus Cepat':**")
    st.bar_chart(df['Lulus Cepat'].value_counts())

    st.write("**Korelasi antar variabel numerik:**")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Navigasi halaman
if page == "EDA":
    EDA.run()
