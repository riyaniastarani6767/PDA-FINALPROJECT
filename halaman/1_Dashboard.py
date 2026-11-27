"""Halaman Dashboard: ringkasan aktivitas prediksi, kinerja model, data, dan fitur paling berpengaruh."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from basis_data.database import baca_riwayat
from tampilan import grafik, sumber
from tampilan.gaya import kaki, kartu_angka, kepala, tautan

METRIK = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1-Score"}
METODE_LABEL = {"sample": "Contoh dataset", "manual": "Input manual", "csv": "Unggah CSV"}

kepala(
    "Dashboard",
    "Ringkasan sistem klasifikasi tumor payudara: aktivitas prediksi, kinerja <b>Random Forest</b> dan "
    "<b>SVM</b>, data penelitian, serta fitur yang paling berpengaruh menurut <b>Feature Importance</b> dan <b>SHAP</b>.",
    lencana="RANDOM FOREST · SVM · SHAP",
)

ev = sumber.hasil_evaluasi()
df = sumber.dataset()
try:
    riwayat = baca_riwayat()
except Exception:
    riwayat = pd.DataFrame()

# ---------------------------------------------------------------- KPI
k = st.columns(4)
with k[0]:
    n_hari_ini = 0
    if not riwayat.empty:
        w = pd.to_datetime(riwayat["Tanggal dan waktu"], utc=True, errors="coerce").dt.tz_convert("Asia/Jakarta")
        n_hari_ini = int((w.dt.date == pd.Timestamp.now(tz="Asia/Jakarta").date()).sum())
    kartu_angka("Total prediksi tersimpan", f"{len(riwayat)}", f"{n_hari_ini} prediksi hari ini")
with k[1]:
    n_m = int((riwayat["Hasil klasifikasi"] == "Malignant").sum()) if not riwayat.empty else 0
    kartu_angka("Hasil Malignant", f"{n_m}", f"{n_m / len(riwayat):.1%} dari seluruh prediksi" if len(riwayat) else
                "belum ada prediksi", kuning=True)
if ev:
    cv = {r["model"]: r for r in ev["kfold"]}
    with k[2]:
        kartu_angka("Akurasi SVM (10-Fold ×5)", f"{cv['SVM RBF']['accuracy_mean']:.2%}",
                    f"Random Forest {cv['Random Forest']['accuracy_mean']:.2%}")
    with k[3]:
        kartu_angka("Recall SVM — model terbaik", f"{cv['SVM RBF']['recall_mean']:.2%}",
                    f"Random Forest {cv['Random Forest']['recall_mean']:.2%}", kuning=True)

b1, b2, b3, b4 = st.columns(4)
with b1:
    tautan("halaman/2_Prediksi.py", "Prediksi baru", ":material/biotech:")
with b2:
    tautan("halaman/3_Riwayat_Prediksi.py", "Lihat riwayat", ":material/history:")
with b3:
    tautan("halaman/4_Evaluasi_Model.py", "Detail evaluasi", ":material/analytics:")
with b4:
    tautan("halaman/5_Tentang_Data.py", "Tentang data", ":material/dataset:")

# ---------------------------------------------------------------- Aktivitas prediksi
st.markdown("### Aktivitas prediksi")
if riwayat.empty:
    st.info("Belum ada prediksi tersimpan. Grafik aktivitas akan muncul setelah prediksi pertama dijalankan.")
else:
    waktu = pd.to_datetime(riwayat["Tanggal dan waktu"], utc=True, errors="coerce").dt.tz_convert("Asia/Jakarta")
    data_tren = pd.DataFrame({"tanggal": waktu.dt.date, "hasil": riwayat["Hasil klasifikasi"]})
    a1, a2 = st.columns([1.7, 1])
    with a1:
        with st.container(border=True):
            st.altair_chart(grafik.tren_harian(data_tren, "Jumlah Prediksi per Hari (14 hari terakhir)"), use_container_width=True)
    with a2:
        with st.container(border=True):
            komposisi = riwayat["Hasil klasifikasi"].value_counts()
            st.altair_chart(grafik.donat({"Benign": int(komposisi.get("Benign", 0)),
                                    "Malignant": int(komposisi.get("Malignant", 0))}, "Komposisi Hasil Prediksi"),
                      use_container_width=True)
    c1, c2 = st.columns([1, 1.7])
    with c1:
        with st.container(border=True):
            st.markdown("**Prediksi per model**")
            per_model = riwayat.groupby(["Model", "Hasil klasifikasi"]).size().unstack(fill_value=0)
            per_model["Total"] = per_model.sum(axis=1)
            st.dataframe(per_model, use_container_width=True)
            st.markdown("**Metode input**")
            st.dataframe(riwayat["Metode input"].map(lambda v: METODE_LABEL.get(v, v)).value_counts()
                         .rename_axis("Metode").rename("Jumlah").to_frame(), use_container_width=True)
    with c2:
        with st.container(border=True):
            st.markdown("**5 prediksi terakhir**")
            terakhir = riwayat.head(5).assign(Waktu=waktu.head(5).dt.strftime("%d-%m-%Y %H:%M"))
            st.dataframe(
                terakhir[["ID", "Waktu", "Model", "Hasil klasifikasi", "Probabilitas Malignant"]],
                hide_index=True, use_container_width=True,
                column_config={"Probabilitas Malignant": st.column_config.ProgressColumn(
                    "Probabilitas Malignant", min_value=0.0, max_value=1.0, format="%.3f")},
            )

# ---------------------------------------------------------------- Kinerja model
if ev:
    st.markdown("### Kinerja model")
    m1, m2 = st.columns([1.7, 1])
    with m1:
        with st.container(border=True):
            tabel_cv = pd.DataFrame({m: [cv[m][f"{x}_mean"] * 100 for x in METRIK] for m in cv}, index=METRIK.values())
            st.altair_chart(grafik.bandingan_metrik(tabel_cv, "Rata-rata Metrik — Stratified 10-Fold ×5"), use_container_width=True)
    with m2:
        with st.container(border=True):
            h = ev["holdout"]["SVM RBF"]
            st.altair_chart(grafik.confusion_matrix(h["TN"], h["FP"], h["FN"], h["TP"], "Confusion Matrix SVM\n(data uji, n = 114)"),
                      use_container_width=True)
    st.caption(f":material/workspace_premium: Model terbaik: **{ev['model_terbaik']['model']}** — {ev['model_terbaik']['alasan']}")

# ---------------------------------------------------------------- Data & fitur
st.markdown("### Data dan fitur paling berpengaruh")
d1, d2, d3 = st.columns([1, 1, 1])
with d1:
    with st.container(border=True):
        n_b, n_m_data = int((df.diagnosis == "B").sum()), int((df.diagnosis == "M").sum())
        st.altair_chart(grafik.donat({"Benign": n_b, "Malignant": n_m_data}, f"Dataset ({len(df)} pasien, 30 fitur)"),
                  use_container_width=True)
fi = sumber.feature_importance()
sg = sumber.shap_global()
with d2:
    with st.container(border=True):
        if fi is not None:
            top = fi.head(5)
            st.altair_chart(grafik.batang_horizontal(top["nama"], top["importance"], "Top 5 — Feature Importance (RF)",
                                               "Importance", desimal=3), use_container_width=True)
with d3:
    with st.container(border=True):
        if sg is not None:
            top = sg["ranking"].head(5)
            st.altair_chart(grafik.batang_horizontal(top["nama"], top["mean_abs_shap"], "Top 5 — SHAP (SVM)",
                                               "Rata-rata |SHAP|", warna=grafik.CM_KUNING, desimal=3), use_container_width=True)

st.caption("Data penelitian: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository. Nilai 30 fitur berasal "
           "dari analisis citra sampel FNA; sistem tidak mengolah citra.")
kaki()
