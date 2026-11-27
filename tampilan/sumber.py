"""Pemuat artefak hasil training (di-cache) dan perhitungan SHAP lokal untuk aplikasi.

Aplikasi TIDAK melatih ulang model: semua angka evaluasi, feature importance, dan SHAP global
dibaca dari folder models/ yang dihasilkan train.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from preprocessing.data import FITUR, KARAKTERISTIK, label_fitur

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATASET = ROOT / "data" / "breast_cancer.csv"

NAMA_MODEL = {"Random Forest": "Random Forest", "SVM RBF": "Support Vector Machine"}
MODEL_UI = ["Random Forest", "Support Vector Machine"]

# Arti 10 karakteristik inti sel (Wolberg dkk., 1993) dalam Bahasa Indonesia
ARTI_KARAKTERISTIK = {
    "radius": ("Jari-jari", "Rata-rata jarak dari titik pusat ke tepi inti sel."),
    "texture": ("Tekstur", "Simpangan baku nilai keabuan (gray-scale) pada citra inti sel."),
    "perimeter": ("Keliling", "Panjang keliling (batas) inti sel."),
    "area": ("Luas", "Luas penampang inti sel."),
    "smoothness": ("Kehalusan", "Variasi lokal panjang jari-jari; makin kecil makin halus tepinya."),
    "compactness": ("Kekompakan", "Keliling² / luas − 1,0; menggambarkan kepadatan bentuk."),
    "concavity": ("Kecekungan", "Tingkat keparahan bagian cekung pada kontur inti sel."),
    "concave_points": ("Titik cekung", "Jumlah bagian cekung pada kontur inti sel."),
    "symmetry": ("Simetri", "Kesimetrisan bentuk inti sel."),
    "fractal_dimension": ("Dimensi fraktal", "Kerumitan garis tepi (coastline approximation − 1)."),
}
ARTI_VARIASI = {
    "mean": "nilai rata-rata seluruh inti sel pada citra",
    "se": "standard error (variasi) antar inti sel",
    "worst": "rata-rata 3 nilai terbesar (inti sel paling abnormal)",
}


def arti_fitur(nama: str) -> str:
    k, v = nama.rsplit("_", 1)
    indo, arti = ARTI_KARAKTERISTIK[k]
    return f"{indo} — {arti} Variasi: {ARTI_VARIASI[v]}."


@st.cache_data(show_spinner=False)
def dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df["diagnosis"] = df["diagnosis"].astype(str).str.upper().str.strip()
    df["label"] = df["diagnosis"].map({"M": "Malignant", "B": "Benign"})
    return df


@st.cache_data(show_spinner=False)
def rentang_fitur() -> pd.DataFrame:
    df = dataset()
    return pd.DataFrame({
        "min": df[FITUR].min(), "median": df[FITUR].median(), "max": df[FITUR].max(),
        "mean_benign": df.loc[df.diagnosis == "B", FITUR].mean(),
        "mean_malignant": df.loc[df.diagnosis == "M", FITUR].mean(),
    })


def _json(nama: str) -> dict | None:
    path = MODELS / nama
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def hasil_evaluasi() -> dict | None:
    return _json("hasil_evaluasi.json")


@st.cache_data(show_spinner=False)
def hasil_eda() -> dict | None:
    return _json("hasil_eda.json")


@st.cache_data(show_spinner=False)
def hasil_preprocessing() -> dict | None:
    return _json("hasil_preprocessing.json")


@st.cache_data(show_spinner=False)
def kfold() -> pd.DataFrame | None:
    path = MODELS / "hasil_kfold.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def feature_importance() -> pd.DataFrame | None:
    path = MODELS / "feature_importance_rf.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def shap_global() -> dict | None:
    path = MODELS / "shap_global.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def _svm():
    return joblib.load(MODELS / "model_svm.pkl")


@st.cache_resource(show_spinner=False)
def _explainer():
    from interpretasi_shap import shap_svm

    background = joblib.load(MODELS / "shap_background.pkl")
    return shap_svm.buat_explainer(_svm()["model"], background)


def shap_tersedia() -> bool:
    return (MODELS / "shap_background.pkl").exists() and (MODELS / "model_svm.pkl").exists()


@st.cache_data(show_spinner=False, max_entries=500)
def shap_lokal(nilai: tuple) -> dict:
    """SHAP lokal model SVM untuk satu pasien (nilai = 30 fitur asli berurutan sesuai FITUR)."""
    from interpretasi_shap import shap_svm

    x = np.asarray(nilai, dtype=float).reshape(1, -1)
    x_scaled = _svm()["scaler"].transform(x)
    hasil = shap_svm.jelaskan_lokal(_explainer(), x_scaled)
    return {k: (np.asarray(v).tolist() if k == "shap" else v) for k, v in hasil.items()}


@st.cache_data(show_spinner=False)
def prob_svm_data_uji() -> list:
    data = shap_global()
    X = _svm()["scaler"].transform(np.asarray(data["X_asli"], dtype=float))
    return np.asarray(_svm()["model"].predict_proba(X))[:, 1].tolist()


def tabel_shap(nilai_shap, nilai_asli) -> pd.DataFrame:
    df = pd.DataFrame({
        "Fitur": [label_fitur(f) for f in FITUR],
        "Nilai fitur": np.asarray(nilai_asli, dtype=float),
        "Nilai SHAP": np.asarray(nilai_shap, dtype=float),
    })
    df["Arah pengaruh"] = np.where(df["Nilai SHAP"] > 0, "→ Malignant", np.where(df["Nilai SHAP"] < 0, "→ Benign", "netral"))
    return df.reindex(df["Nilai SHAP"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def kalimat_shap(tabel: pd.DataFrame, n: int = 3) -> str:
    """Ringkasan bahasa sederhana dari hasil SHAP lokal."""
    ke_m = tabel[tabel["Nilai SHAP"] > 0].head(n)
    ke_b = tabel[tabel["Nilai SHAP"] < 0].head(n)
    bagian = []
    if len(ke_m):
        bagian.append("Fitur yang paling <b>mendorong ke Malignant</b>: "
                      + ", ".join(f"<b>{r.Fitur}</b> ({r._2:.4g})" for r in ke_m.itertuples()) + ".")
    if len(ke_b):
        bagian.append("Fitur yang paling <b>mendorong ke Benign</b>: "
                      + ", ".join(f"<b>{r.Fitur}</b> ({r._2:.4g})" for r in ke_b.itertuples()) + ".")
    return " ".join(bagian) if bagian else "Tidak ada fitur yang berpengaruh."


def daftar_karakteristik() -> pd.DataFrame:
    return pd.DataFrame([
        {"No": i + 1, "Karakteristik": k.replace("_", " ").title(), "Nama Indonesia": ARTI_KARAKTERISTIK[k][0],
         "Penjelasan": ARTI_KARAKTERISTIK[k][1], "Kolom": f"{k}_mean, {k}_se, {k}_worst"}
        for i, k in enumerate(KARAKTERISTIK)
    ])


def template_csv() -> bytes:
    """Contoh file CSV: 2 baris pasien dari dataset (1 Benign, 1 Malignant) dengan 30 kolom fitur."""
    df = dataset()
    contoh = pd.concat([df[df.diagnosis == "B"].head(1), df[df.diagnosis == "M"].head(1)])
    return contoh[FITUR].to_csv(index=False).encode("utf-8")


def contoh_csv_dataset(n: int = 20) -> bytes:
    df = dataset()
    contoh = pd.concat([df[df.diagnosis == "B"].head(n // 2), df[df.diagnosis == "M"].head(n - n // 2)])
    return contoh[["id", "diagnosis"] + FITUR].to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Perhitungan EDA untuk tab Eksplorasi Data (memakai fungsi preprocessing/eda.py yang sama dengan train.py)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def eda_statistik() -> pd.DataFrame:
    from preprocessing import eda
    return eda.statistik_deskriptif(dataset())


@st.cache_data(show_spinner=False)
def eda_outlier() -> pd.DataFrame:
    from preprocessing import eda
    return eda.deteksi_outlier(dataset())[0]


@st.cache_data(show_spinner=False)
def eda_korelasi_target() -> pd.Series:
    from preprocessing import eda
    return eda.korelasi_target(dataset())


@st.cache_data(show_spinner=False)
def eda_rasio_kelas() -> pd.DataFrame:
    from preprocessing import eda
    return eda.rasio_rata_kelas(dataset())


@st.cache_data(show_spinner=False)
def eda_pasangan_korelasi() -> pd.DataFrame:
    from preprocessing import eda
    return eda.pasangan_korelasi_tinggi(dataset())


@st.cache_data(show_spinner=False)
def eda_matriks_korelasi(variasi: str) -> pd.DataFrame:
    from preprocessing.data import KARAKTERISTIK
    kolom = [f"{k}_{variasi}" for k in KARAKTERISTIK]
    kor = dataset()[kolom].corr()
    kor.index = kor.columns = [ARTI_KARAKTERISTIK[k][0] for k in KARAKTERISTIK]
    return kor
