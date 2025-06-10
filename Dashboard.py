import streamlit as st

# Sidebar navigasi
st.sidebar.title("📊 Menu")
page = st.sidebar.radio("Pilih Halaman:", ["EDA", "Modeling", "Prediksi"])

# Halaman utama
st.title("🎓 Dashboard Prediksi Kelulusan Mahasiswa")
st.markdown("""
Gunakan sidebar untuk berpindah antar halaman:
- **EDA**: Eksplorasi Data
- **Modeling**: Pelatihan Model
- **Prediksi**: Input data dan lihat hasil prediksi
""")

# Navigasi antar halaman
if page == "EDA":
    import pages.EDA as eda
    eda.run()

elif page == "Modeling":
    import pages.Modeling as modeling
    modeling.run()

elif page == "Prediksi":
    import pages.Predict as predict
    predict.run()
