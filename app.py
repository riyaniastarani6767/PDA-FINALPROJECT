from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
CSS_PATH = ROOT / "assets" / "styles.css"


st.set_page_config(
    page_title="Breast Tumor Classification",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.markdown("### Sistem Klasifikasi Tumor Payudara")
st.markdown(
    "Prototipe penelitian untuk membandingkan **Random Forest** dan **SVM** "
    "dengan interpretasi model berbasis Feature Importance dan SHAP."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="status-banner">', unsafe_allow_html=True)
st.markdown("**Status:** Kerangka antarmuka siap; backend model dan database belum diaktifkan.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Navigasi</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/01_Dashboard.py", label="Dashboard", icon="🏠")
    st.page_link("pages/02_Prediksi.py", label="Prediksi", icon="🧪")
    st.page_link("pages/03_Riwayat.py", label="Riwayat Prediksi", icon="🗂️")
with col2:
    st.page_link("pages/04_Evaluasi.py", label="Evaluasi & Interpretasi", icon="📊")
    st.page_link("pages/05_Dataset.py", label="Informasi Dataset", icon="📋")

st.caption("Aplikasi ini merupakan prototipe penelitian dan edukasi, bukan alat diagnosis klinis.")
st.markdown('</div>', unsafe_allow_html=True)
