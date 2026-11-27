"""Aplikasi Sistem Klasifikasi Tumor Payudara (Random Forest & SVM + SHAP).

Jalankan dengan:  streamlit run app.py
Halaman sesuai rancangan antarmuka laporan (Subbab 3.4.5): Dashboard, Prediksi, Riwayat Prediksi,
Evaluasi Model, dan Tentang Data.
"""
import streamlit as st

from tampilan.gaya import terapkan_tema

st.set_page_config(
    page_title="Klasifikasi Tumor Payudara",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

halaman = st.navigation(
    {
        "Menu": [
            st.Page("halaman/1_Dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
            st.Page("halaman/2_Prediksi.py", title="Prediksi", icon=":material/biotech:"),
            st.Page("halaman/3_Riwayat_Prediksi.py", title="Riwayat Prediksi", icon=":material/history:"),
            st.Page("halaman/4_Evaluasi_Model.py", title="Evaluasi Model", icon=":material/analytics:"),
            st.Page("halaman/5_Tentang_Data.py", title="Tentang Data", icon=":material/dataset:"),
        ]
    }
)

with st.sidebar:
    st.markdown(
        '<div class="merek"><div class="merek-logo">'
        '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg></div>'
        '<div><div class="merek-nama">Klasifikasi Tumor Payudara</div>'
        '<div class="merek-sub">Random Forest · SVM · SHAP</div></div></div>',
        unsafe_allow_html=True,
    )
terapkan_tema()

halaman.run()
