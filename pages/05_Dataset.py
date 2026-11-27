from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import (
    DatasetLoadError,
    muat_dataset_ui,
    ringkasan_dataset,
    statistik_fitur,
    tabel_daftar_fitur,
)


ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "assets" / "styles.css"
st.set_page_config(page_title="Informasi Dataset", page_icon="📋", layout="wide")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def muat_data_dataset():
    return muat_dataset_ui()


st.title("Informasi Dataset")

try:
    df = muat_data_dataset()
    ringkasan = ringkasan_dataset(df)

    st.markdown('<div class="section-heading">Informasi Umum</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Nama dataset", "Breast Cancer Wisconsin")
    with col2:
        st.metric("Jumlah baris", f"{ringkasan['total_sampel']:,}")
    with col3:
        st.metric("Jumlah kolom", str(ringkasan["jumlah_kolom"]))
    with col4:
        st.metric("Fitur prediktor", str(ringkasan["total_fitur"]))
    with col5:
        st.metric("Kolom target", "diagnosis")

    st.markdown('<div class="section-heading">Distribusi Kelas</div>', unsafe_allow_html=True)
    kiri, kanan = st.columns([2, 1])
    distribusi = pd.DataFrame(
        {
            "Kelas": ["Benign", "Malignant"],
            "Jumlah": [ringkasan["benign"], ringkasan["malignant"]],
        }
    ).set_index("Kelas")

    with kiri:
        st.bar_chart(distribusi)
    with kanan:
        st.metric(
            "Benign",
            f"{ringkasan['benign']:,}",
            f"{ringkasan['persen_benign']:.1f}%",
        )
        st.metric(
            "Malignant",
            f"{ringkasan['malignant']:,}",
            f"{ringkasan['persen_malignant']:.1f}%",
        )

    st.markdown('<div class="section-heading">Daftar 30 Fitur Asli</div>', unsafe_allow_html=True)
    tabel_fitur = tabel_daftar_fitur()
    kategori = st.radio(
        "Kelompokkan berdasarkan",
        options=["Semua", "Mean", "Standard Error", "Worst"],
        horizontal=True,
    )
    if kategori == "Semua":
        fitur_ditampilkan = tabel_fitur
    else:
        fitur_ditampilkan = tabel_fitur[tabel_fitur["Kategori"] == kategori]

    st.dataframe(
        fitur_ditampilkan,
        hide_index=True,
        use_container_width=True,
        height=420,
    )

    st.markdown('<div class="section-heading">Statistik Deskriptif Fitur</div>', unsafe_allow_html=True)
    stat = statistik_fitur(df)
    st.dataframe(
        stat.style.format(
            {
                "Mean": "{:.4f}",
                "Median": "{:.4f}",
                "Standard deviation": "{:.4f}",
                "Minimum": "{:.4f}",
                "Maximum": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=520,
    )

    st.markdown('<div class="section-heading">Preview Dataset</div>', unsafe_allow_html=True)
    jumlah_baris = st.selectbox("Jumlah baris", options=[10, 25, 50, 100], index=0)
    kolom_preview = ["diagnosis"] + tabel_fitur["Nama fitur asli"].tolist()
    st.dataframe(
        df[kolom_preview].head(jumlah_baris),
        hide_index=True,
        use_container_width=True,
        height=390,
    )

except DatasetLoadError as exc:
    st.error(str(exc))

st.markdown("<div class=\"disclaimer\">", unsafe_allow_html=True)
st.markdown(
    "Seluruh angka pada halaman ini dihitung dari data/breast_cancer.csv. "
    "Kolom ID tidak digunakan sebagai fitur prediktor."
)
st.markdown("</div>", unsafe_allow_html=True)
