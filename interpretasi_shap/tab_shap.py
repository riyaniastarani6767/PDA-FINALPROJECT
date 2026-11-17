"""
Komponen tampilan SHAP untuk aplikasi Streamlit.
- F-08: penjelasan SHAP satu pasien (grafik batang + tabel fitur beserta arah pengaruhnya)
- F-11: penjelasan SHAP global (summary plot + urutan fitur paling berpengaruh)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from interpretasi_shap import shap_svm
from preprocessing.data import FITUR


def tampilkan_shap_lokal(hasil_lokal: dict, nilai_asli, top: int = 10, kunci: str = "lokal"):
    nilai_shap = np.asarray(hasil_lokal["shap"], dtype=float)
    fig = shap_svm.plot_lokal(nilai_shap, nilai_asli, top=top,
                              judul=f"{top} Fitur dengan Kontribusi Terbesar (SVM RBF)")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    base, pred = hasil_lokal["base_value"], hasil_lokal["prediksi"]
    st.caption(
        f"Nilai dasar (rata-rata probabilitas Malignant pada background data) = **{base:.3f}**. "
        f"Nilai dasar + jumlah seluruh nilai SHAP = **{base + nilai_shap.sum():.3f}**, "
        f"sama dengan probabilitas Malignant dari model = **{pred:.3f}** (sifat *local accuracy*)."
    )
    tabel = shap_svm.tabel_lokal(nilai_shap, nilai_asli)
    st.dataframe(
        tabel.head(top).style.format({"Nilai fitur": "{:.4f}", "Nilai SHAP": "{:+.4f}"}),
        hide_index=True, width="stretch",
    )
    with st.expander("Lihat nilai SHAP seluruh 30 fitur"):
        st.dataframe(tabel.style.format({"Nilai fitur": "{:.4f}", "Nilai SHAP": "{:+.4f}"}),
                     hide_index=True, width="stretch", key=f"tabel_{kunci}")


def tampilkan_shap_global(data_global: dict, top: int = 15):
    kiri, kanan = st.columns([3, 2])
    with kiri:
        fig = shap_svm.plot_summary(data_global["shap"], data_global["X_asli"], top=top)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Setiap titik adalah satu pasien data uji. Warna merah = nilai fitur tinggi, biru = rendah. "
                   "Titik di kanan garis nol mendorong prediksi ke Malignant, di kiri ke Benign.")
    with kanan:
        ranking = data_global["ranking"][["peringkat", "nama", "mean_abs_shap"]].rename(
            columns={"peringkat": "No", "nama": "Fitur", "mean_abs_shap": "Rata-rata |SHAP|"})
        st.markdown("**Urutan fitur paling berpengaruh (SVM RBF)**")
        st.dataframe(ranking.head(top).style.format({"Rata-rata |SHAP|": "{:.4f}"}),
                     hide_index=True, width="stretch", height=560)


def nilai_shap_ke_dict(nilai_shap) -> dict:
    return {f: float(v) for f, v in zip(FITUR, np.asarray(nilai_shap, dtype=float))}
