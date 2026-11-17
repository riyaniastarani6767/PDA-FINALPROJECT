"""
Interpretasi model SVM dengan SHAP KernelExplainer
(Laporan Subbab 2.1.9, 2.1.9.1, 2.1.9.2, 3.5.3 – Flowchart Algoritma SHAP) – menjawab Rumusan Masalah 2.

Tahapan (Flowchart SHAP):
1. Input & prediksi awal: data pasien (sudah distandardisasi) diprediksi model SVM RBF.
2. Inisialisasi KernelExplainer: background data = data latih yang diringkas
   menjadi 20 titik dengan k-means, sebagai pembanding saat fitur dianggap "tidak ada".
3. Simulasi kombinasi (coalition): sebagian fitur diganti nilai background,
   lalu perubahan prediksi diamati.
4. Perhitungan Shapley value dengan regresi linear berbobot (aproksimasi).
   Sifat local accuracy: f(x) = phi_0 + sum(phi_i)  (Persamaan 2.8).
5. Interpretasi arah: SHAP positif mendorong ke Malignant, negatif ke Benign.
6. Keluaran: prediksi + besar kontribusi setiap fitur (lokal per pasien dan global).

Output model yang dijelaskan adalah probabilitas Malignant dari SVM (Platt scaling).
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocessing.data import FITUR, label_fitur

JUMLAH_BACKGROUND = 20
SEED = 42  # agar hasil sampling coalition SHAP dapat diulang (reproducible)
MERAH, HIJAU = "#D97706", "#3D6F0C"


def buat_background(X_latih_scaled, k: int = JUMLAH_BACKGROUND):
    """Langkah 2: ringkas data latih menjadi k titik dengan k-means."""
    return shap.kmeans(np.asarray(X_latih_scaled, dtype=float), k)


def buat_explainer(model_svm, background):
    """KernelExplainer hanya membutuhkan fungsi prediksi model (model-agnostic)."""
    def fungsi_prediksi(X):
        return model_svm.predict_proba(X)[:, 1]
    return shap.KernelExplainer(fungsi_prediksi, background)


def jelaskan_lokal(explainer, x_scaled, nsamples="auto") -> dict:
    """SHAP untuk satu pasien (local explanation) + cek local accuracy."""
    x = np.asarray(x_scaled, dtype=float).reshape(1, -1)
    np.random.seed(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nilai = np.asarray(explainer.shap_values(x, nsamples=nsamples, l1_reg=False, silent=True)).reshape(-1)
    phi0 = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    fx = float(explainer.model.f(x).reshape(-1)[0])
    return {
        "shap": nilai,
        "base_value": phi0,
        "prediksi": fx,
        "local_accuracy": abs(phi0 + nilai.sum() - fx) < 1e-6,
    }


def jelaskan_global(explainer, X_scaled, nsamples=1000) -> np.ndarray:
    """SHAP untuk banyak pasien (global explanation). Baris = pasien, kolom = fitur."""
    np.random.seed(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nilai = explainer.shap_values(np.asarray(X_scaled, dtype=float), nsamples=nsamples, l1_reg=False, silent=True)
    return np.asarray(nilai)


def tabel_lokal(nilai_shap, nilai_asli, fitur=FITUR) -> pd.DataFrame:
    df = pd.DataFrame({
        "Fitur": [label_fitur(f) for f in fitur],
        "Nilai fitur": np.asarray(nilai_asli, dtype=float),
        "Nilai SHAP": np.asarray(nilai_shap, dtype=float),
    })
    df["Arah pengaruh"] = np.where(df["Nilai SHAP"] > 0, "→ Malignant",
                                   np.where(df["Nilai SHAP"] < 0, "→ Benign", "netral"))
    return df.reindex(df["Nilai SHAP"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def ranking_global(matriks_shap, fitur=FITUR) -> pd.DataFrame:
    """Urutan fitur berdasarkan rata-rata |SHAP| (Rumusan Masalah 2 untuk SVM)."""
    df = pd.DataFrame({"fitur": fitur, "mean_abs_shap": np.abs(matriks_shap).mean(axis=0)})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.insert(0, "peringkat", range(1, len(df) + 1))
    df["nama"] = df["fitur"].map(label_fitur)
    return df


def plot_lokal(nilai_shap, nilai_asli, fitur=FITUR, top=10, judul=None, path=None):
    df = tabel_lokal(nilai_shap, nilai_asli, fitur).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * top + 1.3))
    warna = [MERAH if v > 0 else HIJAU for v in df["Nilai SHAP"]]
    label = [f"{f} = {v:.3g}" for f, v in zip(df["Fitur"], df["Nilai fitur"])]
    ax.barh(label, df["Nilai SHAP"], color=warna)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("Nilai SHAP (kontribusi terhadap probabilitas Malignant)")
    ax.set_title(judul or f"{top} Fitur dengan Kontribusi Terbesar")
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MERAH, label="Mendorong ke Malignant"),
                       Patch(color=HIJAU, label="Mendorong ke Benign")], loc="lower right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_summary(matriks_shap, X_asli, fitur=FITUR, top=15, path=None):
    """SHAP summary plot (global explanation)."""
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        shap.summary_plot(np.asarray(matriks_shap), pd.DataFrame(np.asarray(X_asli), columns=[label_fitur(f) for f in fitur]),
                          max_display=top, show=False, plot_size=(8, 0.4 * top + 1.5))
    fig = plt.gcf()
    fig.suptitle("SHAP Summary Plot – SVM RBF", fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_global_bar(ranking: pd.DataFrame, top=10, path=None):
    data = ranking.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * top + 1.2))
    ax.barh(data["nama"], data["mean_abs_shap"], color=HIJAU)
    for i, v in enumerate(data["mean_abs_shap"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Rata-rata |nilai SHAP|")
    ax.set_title(f"{top} Fitur Paling Berpengaruh – SVM RBF (SHAP)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
