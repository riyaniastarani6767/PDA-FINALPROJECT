from pathlib import Path

import streamlit as st

from components.cards import tampilkan_metric_card
from utils.data_loader import (
    DatasetLoadError,
    muat_dataset_ui,
    ringkasan_dataset,
    status_artefak_model,
)


ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "assets" / "styles.css"
st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def muat_data_dashboard():
    return muat_dataset_ui()


st.title("Breast Cancer Classification System")
st.subheader("Sistem Klasifikasi Tumor Payudara Berdasarkan Karakteristik Inti Sel")
st.markdown(
    "Prototipe penelitian ini membandingkan Random Forest dan Support Vector Machine "
    "untuk klasifikasi tumor berdasarkan 30 karakteristik inti sel, serta menyiapkan "
    "interpretasi model melalui Feature Importance dan SHAP."
)

try:
    df = muat_data_dashboard()
    ringkasan = ringkasan_dataset(df)

    st.markdown('<div class="section-heading">Ringkasan Dataset</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tampilkan_metric_card("Total sampel", f"{ringkasan['total_sampel']:,}")
    with col2:
        tampilkan_metric_card("Total fitur prediktor", str(ringkasan["total_fitur"]))
    with col3:
        tampilkan_metric_card("Kelas Benign", f"{ringkasan['benign']:,}")
    with col4:
        tampilkan_metric_card("Kelas Malignant", f"{ringkasan['malignant']:,}")

except DatasetLoadError as exc:
    st.error(str(exc))
    ringkasan = None

st.markdown('<div class="section-heading">Status Model</div>', unsafe_allow_html=True)
status_model = status_artefak_model()
col_rf, col_svm = st.columns(2)
with col_rf:
    st.metric("Random Forest", status_model["Random Forest"])
with col_svm:
    st.metric("Support Vector Machine", status_model["Support Vector Machine"])

st.caption(
    "Status diperoleh hanya dari keberadaan artefak .pkl. Belum ada klaim bahwa model "
    "atau preprocessing inference telah lulus validasi backend."
)

st.markdown('<div class="section-heading">Informasi Penelitian</div>', unsafe_allow_html=True)
left, right = st.columns([1, 1])
with left:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("**Tujuan sistem**")
    st.write(
        "Membandingkan kemampuan Random Forest dan SVM dalam mengklasifikasikan "
        "tumor sebagai Benign atau Malignant berdasarkan karakteristik inti sel."
    )
    st.markdown("**Dataset**")
    st.write(
        "Breast Cancer Wisconsin Diagnostic, dengan 30 fitur numerik yang berasal "
        "dari mean, standard error, dan worst dari sepuluh karakteristik inti sel."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("**Algoritma yang diteliti**")
    st.write("Random Forest dan Support Vector Machine dengan kernel RBF.")
    st.markdown("**Metrik evaluasi**")
    st.write("Accuracy, Precision, Recall, dan F1-score; positif adalah Malignant.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
st.markdown(
    "Sistem ini merupakan prototipe penelitian dan bukan alat diagnosis medis. "
    "Hasil klasifikasi tidak dapat menggantikan pemeriksaan dan keputusan tenaga medis profesional."
)
st.markdown('</div>', unsafe_allow_html=True)
