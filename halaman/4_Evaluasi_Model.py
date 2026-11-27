"""Halaman Evaluasi Model: kinerja RF vs SVM, confusion matrix, Stratified 10-Fold, Feature Importance, dan SHAP.

Seluruh angka dibaca dari artefak hasil train.py (folder models/); halaman ini tidak melatih ulang model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from preprocessing.data import FITUR, label_fitur
from tampilan import grafik, sumber
from tampilan.gaya import catatan, kaki, kartu_angka, kepala

METRIK = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1-Score"}
NAMA = {"Random Forest": "Random Forest", "SVM RBF": "SVM RBF"}

kepala(
    "Evaluasi Model",
    "Perbandingan kinerja Random Forest dan SVM (kernel RBF) pada data uji hold-out 80:20 dan "
    "Stratified 10-Fold Cross Validation (5 kali ulangan), beserta interpretasi model dengan "
    "Feature Importance dan SHAP.",
)

ev = sumber.hasil_evaluasi()
if ev is None:
    st.error("Hasil evaluasi belum tersedia (models/hasil_evaluasi.json). Jalankan `python train.py` terlebih dahulu.")
    st.stop()

holdout, cv = ev["holdout"], {r["model"]: r for r in ev["kfold"]}
terbaik = ev["model_terbaik"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":material/insights: Ringkasan kinerja", ":material/grid_view: Confusion matrix", ":material/repeat: Stratified 10-Fold",
    ":material/account_tree: Feature Importance (RF)", ":material/manage_search: SHAP (SVM)",
])

# ---------------------------------------------------------------- Ringkasan
with tab1:
    k = st.columns(4)
    for kol, (kunci, nama) in zip(k, METRIK.items()):
        with kol:
            rf, svm = cv["Random Forest"][f"{kunci}_mean"], cv["SVM RBF"][f"{kunci}_mean"]
            kartu_angka(f"{nama} (rata-rata 10-Fold ×5)", f"SVM {svm:.2%}", f"Random Forest {rf:.2%}", kuning=kunci == "recall")
    catatan(f"<b>Model terbaik: {terbaik['model']}</b>. {terbaik['alasan']} Recall diutamakan karena kesalahan "
            "melewatkan pasien Malignant (False Negative) paling berisiko pada konteks medis.")

    kiri, kanan = st.columns([1.25, 1])
    with kiri:
        tabel_cv = pd.DataFrame({NAMA[m]: [cv[m][f"{k}_mean"] * 100 for k in METRIK] for m in cv}, index=METRIK.values())
        st.altair_chart(grafik.bandingan_metrik(tabel_cv, "Rata-rata Metrik — Stratified 10-Fold ×5"), use_container_width=True)
    with kanan:
        st.markdown("**Data uji hold-out (114 pasien)**")
        st.dataframe(pd.DataFrame({
            "Metrik": list(METRIK.values()),
            "Random Forest": [f"{holdout['Random Forest'][k]:.2%}" for k in METRIK],
            "SVM RBF": [f"{holdout['SVM RBF'][k]:.2%}" for k in METRIK],
        }), hide_index=True, use_container_width=True)
        st.markdown("**Stratified 10-Fold ×5 (rata-rata ± simpangan baku)**")
        st.dataframe(pd.DataFrame({
            "Metrik": list(METRIK.values()),
            "Random Forest": [f"{cv['Random Forest'][f'{k}_mean']:.2%} ± {cv['Random Forest'][f'{k}_std']:.2%}" for k in METRIK],
            "SVM RBF": [f"{cv['SVM RBF'][f'{k}_mean']:.2%} ± {cv['SVM RBF'][f'{k}_std']:.2%}" for k in METRIK],
        }), hide_index=True, use_container_width=True)

    with st.expander("Arti setiap metrik"):
        st.markdown(
            "- **Accuracy** — proporsi seluruh prediksi yang benar.\n"
            "- **Precision** — dari semua yang diprediksi Malignant, berapa yang benar-benar Malignant.\n"
            "- **Recall** — dari semua pasien Malignant, berapa yang berhasil terdeteksi (makin tinggi, makin sedikit kasus ganas terlewat).\n"
            "- **F1-Score** — rata-rata harmonik precision dan recall.\n\n"
            "Kelas positif adalah **Malignant**."
        )
    with st.expander("Parameter model"):
        st.json(ev.get("parameter", {}))

# ---------------------------------------------------------------- Confusion matrix
with tab2:
    st.markdown("Confusion matrix pada **data uji hold-out** (114 pasien: 72 Benign, 42 Malignant). "
                "Baris = kelas aktual, kolom = kelas prediksi.")
    c1, c2 = st.columns(2)
    for kol, m in zip((c1, c2), ("Random Forest", "SVM RBF")):
        h = holdout[m]
        with kol:
            st.altair_chart(grafik.confusion_matrix(h["TN"], h["FP"], h["FN"], h["TP"], f"Confusion Matrix {m}\n(data uji, n = 114)"),
                      use_container_width=True)
            st.caption(f"Benar {h['TN'] + h['TP']} dari 114 · salah {h['FP'] + h['FN']} "
                       f"(FP {h['FP']}, FN {h['FN']}).")
    catatan(
        "<b>TN</b> = Benign diprediksi Benign · <b>TP</b> = Malignant diprediksi Malignant · "
        "<b>FP</b> = Benign keliru diprediksi Malignant · <b>FN</b> = Malignant keliru diprediksi Benign (paling berisiko). "
        f"SVM hanya memiliki <b>{holdout['SVM RBF']['FN']} FN</b>, sedangkan Random Forest <b>{holdout['Random Forest']['FN']} FN</b>."
    )

# ---------------------------------------------------------------- K-Fold
with tab3:
    df_cv = sumber.kfold()
    if df_cv is None:
        st.warning("Hasil K-Fold belum tersedia (models/hasil_kfold.csv).")
    else:
        st.markdown(f"Data dibagi menjadi 10 bagian dengan proporsi kelas yang sama; setiap bagian bergantian menjadi "
                    f"data uji. Proses diulang 5 kali dengan pengacakan berbeda sehingga ada **{len(df_cv) // 2} pengujian** "
                    "per model. Sebaran nilai yang sempit menunjukkan model stabil.")
        pilih = st.radio("Metrik", list(METRIK.values()), horizontal=True, index=2, key="metrik_kfold")
        kunci = {v: k for k, v in METRIK.items()}[pilih]
        a, b = st.columns([1, 1])
        with a:
            st.altair_chart(grafik.boxplot_kfold(df_cv, kunci, pilih), use_container_width=True)
        with b:
            ring = df_cv.groupby("model")[list(METRIK)].agg(["mean", "std", "min", "max"])[kunci] * 100
            st.dataframe(ring.rename(columns={"mean": "Rata-rata (%)", "std": "Simpangan baku", "min": "Minimum (%)",
                                              "max": "Maksimum (%)"}).round(2), use_container_width=True)
        with st.expander(f"Lihat {len(df_cv)} baris hasil pengujian"):
            st.dataframe(df_cv, hide_index=True, use_container_width=True)
            st.download_button(":material/download: Unduh hasil K-Fold (CSV)", df_cv.to_csv(index=False).encode("utf-8"),
                               "hasil_kfold.csv", "text/csv")

# ---------------------------------------------------------------- Feature importance
with tab4:
    fi = sumber.feature_importance()
    if fi is None:
        st.warning("Feature importance belum tersedia (models/feature_importance_rf.csv).")
    else:
        n = st.slider("Jumlah fitur ditampilkan", 5, 30, 15, key="n_fi")
        top = fi.head(n)
        st.altair_chart(grafik.batang_horizontal(top["nama"], top["importance"], f"{n} Fitur Terpenting — Random Forest",
                                           "Feature importance (mean decrease impurity)"), use_container_width=True)
        catatan(f"Fitur paling berpengaruh pada Random Forest adalah <b>{fi.iloc[0]['nama']}</b> "
                f"({fi.iloc[0]['importance']:.4f}), diikuti <b>{fi.iloc[1]['nama']}</b> dan <b>{fi.iloc[2]['nama']}</b>. "
                "Nilai importance dihitung dari rata-rata penurunan impurity (Gini) setiap kali fitur dipakai untuk "
                "memecah simpul, lalu dinormalkan sehingga total = 1.")
        with st.expander("Tabel lengkap 30 fitur"):
            st.dataframe(fi[["peringkat", "nama", "fitur", "importance"]].rename(columns={
                "peringkat": "Peringkat", "nama": "Fitur", "fitur": "Nama kolom", "importance": "Importance"}),
                hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- SHAP
with tab5:
    data = sumber.shap_global()
    if data is None:
        st.warning("Hasil SHAP belum tersedia (models/shap_global.pkl).")
    else:
        rank = data["ranking"]
        st.markdown(f"SHAP (KernelExplainer) menjelaskan **probabilitas Malignant** dari model SVM pada "
                    f"**{len(data['X_asli'])} pasien data uji**. Nilai positif mendorong prediksi ke Malignant, "
                    "nilai negatif ke Benign.")
        g1, g2 = st.columns(2)
        with g1:
            top = rank.head(10)
            st.altair_chart(grafik.batang_horizontal(top["nama"], top["mean_abs_shap"], "SHAP Global — 10 Fitur Teratas (SVM)",
                                               "Rata-rata |nilai SHAP|", warna=grafik.CM_KUNING), use_container_width=True)
        with g2:
            st.altair_chart(grafik.shap_summary(data["shap"], data["X_asli"], [label_fitur(f) for f in FITUR], top=10),
                      use_container_width=True)
        st.caption("Summary plot: setiap titik = satu pasien. Warna kuning = nilai fitur tinggi, hijau = rendah. "
                   "Titik di kanan garis nol mendorong ke Malignant.")

        fi = sumber.feature_importance()
        if fi is not None:
            st.markdown("**Perbandingan 10 fitur teratas: Feature Importance (RF) vs SHAP (SVM)**")
            banding = pd.DataFrame({
                "Peringkat": range(1, 11),
                "Random Forest (Feature Importance)": fi.head(10)["nama"].tolist(),
                "SVM (SHAP)": rank.head(10)["nama"].tolist(),
            })
            st.dataframe(banding, hide_index=True, use_container_width=True)
            sama = set(fi.head(10)["fitur"]) & set(rank.head(10)["fitur"])
            catatan(f"<b>{len(sama)} dari 10</b> fitur teratas muncul pada kedua metode, terutama fitur kelompok "
                    "<i>worst</i> dan ukuran/bentuk inti sel (concave points, radius, perimeter, area).")

        st.markdown("### SHAP lokal — penjelasan per pasien data uji")
        prob = np.asarray(sumber.prob_svm_data_uji())
        urut = np.argsort(-prob)
        i = st.selectbox(
            "Pilih pasien data uji",
            options=[int(x) for x in urut],
            format_func=lambda j: f"Pasien uji #{j + 1} — prediksi SVM: "
            f"{'Malignant' if prob[j] >= 0.5 else 'Benign'} (P Malignant = {prob[j]:.3f})",
            key="pasien_uji_shap",
        )
        tabel = sumber.tabel_shap(np.asarray(data["shap"])[i], np.asarray(data["X_asli"])[i])
        catatan(sumber.kalimat_shap(tabel))
        st.altair_chart(grafik.shap_lokal(tabel, top=10, judul=f"SHAP Lokal — Pasien Uji #{i + 1} (SVM)"), use_container_width=True)
        st.caption(f"Nilai dasar {float(data['base_value']):.3f} + jumlah nilai SHAP "
                   f"{float(np.asarray(data['shap'])[i].sum()):+.3f} = {float(data['base_value']) + float(np.asarray(data['shap'])[i].sum()):.3f} "
                   f"(probabilitas Malignant dari SVM = {prob[i]:.3f}).")

kaki()
