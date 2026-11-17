"""
Pipeline pelatihan dan evaluasi model mengikuti tahapan CRISP-DM (Laporan Subbab 3.2).

Jalankan dari folder utama proyek:
    python train.py

Hasil:
- models/  : model Random Forest & SVM terlatih (.pkl), background SHAP, hasil EDA,
             keputusan preprocessing, hasil evaluasi
- gambar/  : grafik EDA, confusion matrix, perbandingan metrik, feature importance, SHAP (bahan BAB IV)
- basis_data/riwayat_prediksi.db : tabel model diisi nilai evaluasi
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from basis_data import database as db
from evaluasi import metrik as ev
from interpretasi_shap import shap_svm
from preprocessing import data as prep
from preprocessing import eda
from random_forest import feature_importance as fi
from random_forest.rf_manual import RandomForestManual
from svm.svm_manual import SVMManual

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
GAMBAR = ROOT / "gambar"

# Parameter model (Subbab 3.4.4)
PARAM_RF = dict(n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt",
                random_state=prep.RANDOM_STATE)
PARAM_SVM = dict(C=1.0, gamma="scale")


def judul(teks):
    print(f"\n{'=' * 70}\n{teks}\n{'=' * 70}")


def main(args):
    MODELS.mkdir(exist_ok=True)
    GAMBAR.mkdir(exist_ok=True)
    PARAM_RF["n_estimators"] = args.n_pohon
    mulai = time.time()

    # ------------------------------------------------------------------
    judul("1. DATA UNDERSTANDING (EDA)")
    df = prep.muat_dataset()
    hasil_eda = eda.jalankan_eda(df, GAMBAR)
    q, d, sk, o = hasil_eda["kualitas"], hasil_eda["distribusi_kelas"], hasil_eda["skala"], hasil_eda["outlier"]
    print(f"  Data           : {q['jumlah_data']} baris x {q['jumlah_kolom']} kolom ({q['jumlah_fitur']} fitur numerik)")
    print(f"  Missing value  : {q['missing_value']}   Duplikat: {q['duplikat']}   Nilai negatif: {q['nilai_negatif']}")
    print(f"  Distribusi     : benign {d['benign']} ({d['persen_benign']}%), malignant {d['malignant']} "
          f"({d['persen_malignant']}%), rasio {d['rasio_mayoritas_minoritas']} : 1")
    print(f"  Skala fitur    : {sk['fitur_nilai_terbesar']} maks {sk['nilai_maks_terbesar']} vs "
          f"{sk['fitur_nilai_terkecil']} maks {sk['nilai_maks_terkecil']}")
    print(f"  Outlier (IQR)  : {o['pasien_dengan_outlier']} pasien ({o['pasien_outlier_malignant']} malignant, "
          f"{o['pasien_outlier_benign']} benign); {o['persen_nilai_outlier_dari_malignant']}% nilai outlier dari malignant")
    print(f"  Korelasi tinggi: {hasil_eda['pasangan_korelasi_tinggi']['jumlah']} pasangan fitur > 0,9")
    print(f"  Fitur paling berkaitan dengan malignant: {', '.join(hasil_eda['korelasi_target_teratas'])}")
    (MODELS / "hasil_eda.json").write_text(json.dumps(hasil_eda, indent=2, ensure_ascii=False))
    print("  Grafik EDA disimpan di gambar/eda_*.png")

    # ------------------------------------------------------------------
    judul("2. DATA PREPARATION (berdasarkan temuan EDA)")
    rencana = prep.rencana_preprocessing(hasil_eda)
    for i, r in enumerate(rencana, 1):
        print(f"  {i}. {r['langkah']:<26} temuan : {r['temuan']}")
        print(f"     {'':<26} tindakan: {r['tindakan']}")
    siap = prep.siapkan_data(df, hasil_eda)
    X, y = siap["X"], siap["y"]
    X_latih, X_uji, y_latih, y_uji = siap["X_latih"], siap["X_uji"], siap["y_latih"], siap["y_uji"]
    Xs_latih, Xs_uji, scaler = siap["Xs_latih"], siap["Xs_uji"], siap["scaler"]
    print("\n  Log preprocessing:")
    for baris in siap["log"]:
        print(f"   - {baris}")
    (MODELS / "hasil_preprocessing.json").write_text(
        json.dumps({"rencana": rencana, "log": siap["log"]}, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    judul("3. MODELING (hold-out 80:20)")
    t = time.time()
    rf = RandomForestManual(**PARAM_RF).fit(Xs_latih, y_latih)
    print(f"  Random Forest : {len(rf.trees)} pohon, Entropy & Information Gain ({time.time() - t:.1f} detik)")
    t = time.time()
    svm = SVMManual(**PARAM_SVM, probability=True).fit(Xs_latih, y_latih)
    print(f"  SVM RBF       : C = {svm.C}, gamma = scale ({svm.gamma_:.5f}), "
          f"{svm.n_support_} support vector, bias b = {svm.intercept_:.4f} ({time.time() - t:.1f} detik)")

    joblib.dump({"model": rf, "scaler": scaler, "fitur": prep.FITUR, "param": PARAM_RF}, MODELS / "model_rf.pkl")
    joblib.dump({"model": svm, "scaler": scaler, "fitur": prep.FITUR, "param": PARAM_SVM}, MODELS / "model_svm.pkl")
    print("  Model disimpan: models/model_rf.pkl, models/model_svm.pkl")

    # ------------------------------------------------------------------
    judul("4. EVALUATION – Rumusan Masalah 1")
    hasil_holdout = {}
    for nama, model in [("Random Forest", rf), ("SVM RBF", svm)]:
        m = ev.hitung_metrik(y_uji, model.predict(Xs_uji))
        hasil_holdout[nama] = m
        print(f"  [Hold-out] {nama:<13} acc {m['accuracy']:.4f}  prec {m['precision']:.4f}  "
              f"rec {m['recall']:.4f}  f1 {m['f1_score']:.4f}  | TP {m['TP']} TN {m['TN']} FP {m['FP']} FN {m['FN']}")
        nama_file = "cm_random_forest.png" if nama == "Random Forest" else "cm_svm.png"
        ev.plot_confusion_matrix(m, f"Confusion Matrix – {nama}\n(data uji hold-out, n = {len(y_uji)})", GAMBAR / nama_file)
        plt.close("all")

    ringkasan_dict, terbaik = None, None
    if not args.tanpa_kfold:
        print("\n  Stratified 10-Fold Cross Validation x 5 (50 pengujian per model)...")
        hasil_kfold = ev.validasi_kfold(
            {"Random Forest": lambda: RandomForestManual(**PARAM_RF),
             "SVM RBF": lambda: SVMManual(**PARAM_SVM)}, X, y)
        hasil_kfold.to_csv(MODELS / "hasil_kfold.csv", index=False)
        ringkasan = ev.ringkas_kfold(hasil_kfold)
        print("\n" + ev.format_ringkasan(ringkasan).to_string(index=False))
        terbaik = ev.pilih_model_terbaik(ringkasan)
        print(f"\n  MODEL TERBAIK: {terbaik['model']} – {terbaik['alasan']}")
        ev.plot_perbandingan_metrik(ringkasan, GAMBAR / "perbandingan_metrik.png")
        plt.close("all")
        ringkasan_dict = ringkasan.to_dict(orient="records")
        for r in ringkasan_dict:
            metode = "Feature Importance" if r["model"] == "Random Forest" else "SHAP KernelExplainer"
            db.simpan_model(r["model"], metode, {m: r[f"{m}_mean"] for m in ev.METRIK})
    else:
        print("  (K-Fold dilewati: --tanpa-kfold)")
        for nama, m in hasil_holdout.items():
            db.simpan_model(nama, "Feature Importance" if nama == "Random Forest" else "SHAP KernelExplainer", m)

    # ------------------------------------------------------------------
    judul("5. INTERPRETASI MODEL – Rumusan Masalah 2")
    tabel_fi = fi.tabel_feature_importance(rf)
    tabel_fi.to_csv(MODELS / "feature_importance_rf.csv", index=False)
    fi.plot_feature_importance(tabel_fi, path=GAMBAR / "feature_importance_rf.png")
    plt.close("all")
    print("  Random Forest – 5 fitur teratas (feature importance):")
    for _, r in tabel_fi.head(5).iterrows():
        print(f"    {r['peringkat']}. {r['nama']:<26} {r['importance']:.4f}")

    t = time.time()
    background = shap_svm.buat_background(Xs_latih)
    explainer = shap_svm.buat_explainer(svm, background)
    matriks = shap_svm.jelaskan_global(explainer, Xs_uji, nsamples=args.nsamples)
    ranking = shap_svm.ranking_global(matriks)
    ranking.to_csv(MODELS / "ranking_shap_svm.csv", index=False)
    shap_svm.plot_summary(matriks, X_uji.to_numpy(), path=GAMBAR / "shap_summary_svm.png")
    plt.close("all")
    shap_svm.plot_global_bar(ranking, path=GAMBAR / "shap_global_svm.png")
    plt.close("all")
    print(f"  SVM – 5 fitur teratas (rata-rata |SHAP|, {len(y_uji)} data uji, {time.time() - t:.0f} detik):")
    for _, r in ranking.head(5).iterrows():
        print(f"    {r['peringkat']}. {r['nama']:<26} {r['mean_abs_shap']:.4f}")

    # Contoh penjelasan lokal: satu pasien malignant dari data uji
    idx = int(np.flatnonzero(y_uji == 1)[0])
    lokal = shap_svm.jelaskan_lokal(explainer, Xs_uji[idx])
    shap_svm.plot_lokal(lokal["shap"], X_uji.iloc[idx].to_numpy(),
                        judul="Contoh SHAP Lokal – Pasien Malignant (data uji)", path=GAMBAR / "shap_lokal_contoh.png")
    plt.close("all")
    print(f"  Local accuracy (phi_0 + sum phi = f(x)): {lokal['local_accuracy']}")

    joblib.dump(background, MODELS / "shap_background.pkl")
    joblib.dump({"shap": matriks, "X_asli": X_uji.to_numpy(), "ranking": ranking,
                 "base_value": float(np.asarray(explainer.expected_value).reshape(-1)[0])},
                MODELS / "shap_global.pkl")

    # ------------------------------------------------------------------
    ringkas = {
        "data": {**hasil_eda["kualitas"], **hasil_eda["distribusi_kelas"]},
        "pembagian": {"latih": int(len(y_latih)), "uji": int(len(y_uji))},
        "parameter": {"Random Forest": PARAM_RF, "SVM RBF": {**PARAM_SVM, "gamma_nilai": svm.gamma_,
                                                           "jumlah_support_vector": svm.n_support_}},
        "holdout": hasil_holdout,
        "kfold": ringkasan_dict,
        "model_terbaik": terbaik,
    }
    (MODELS / "hasil_evaluasi.json").write_text(json.dumps(ringkas, indent=2, ensure_ascii=False))
    judul(f"SELESAI dalam {time.time() - mulai:.0f} detik – jalankan aplikasi: streamlit run app.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Training & evaluasi Random Forest dan SVM (CRISP-DM)")
    p.add_argument("--n-pohon", type=int, default=100, help="jumlah pohon Random Forest (n_estimators)")
    p.add_argument("--tanpa-kfold", action="store_true", help="lewati Stratified 10-Fold x 5 (lebih cepat)")
    p.add_argument("--nsamples", type=int, default=1000, help="jumlah sampel coalition SHAP global per pasien")
    main(p.parse_args())
