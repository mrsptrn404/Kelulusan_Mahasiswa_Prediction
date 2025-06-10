import streamlit as st
from pages import EDA, Modeling, Predict

# Sidebar navigasi
st.sidebar.title("📊 Menu")
page = st.sidebar.radio("Pilih Halaman:", ["EDA", "Modeling", "Predict"])

# Halaman utama
st.title("🎓 Dashboard Prediksi Kelulusan Mahasiswa")
st.markdown("""
Gunakan sidebar untuk berpindah antar halaman:
- **EDA**: Eksplorasi Data
- **Modeling**: Pelatihan Model
- **Predict**: Input data dan lihat hasil prediksi
""")

# Navigasi antar halaman
if page == "EDA":
    import pages.EDA as eda
    eda.run()

elif page == "Modeling":
    import pages.Modeling as modeling
    modeling.run()

elif page == "Predict":
    import pages.Predict as predict
    predict.run()
