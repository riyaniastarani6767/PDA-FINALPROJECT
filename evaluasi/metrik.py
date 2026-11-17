"""
Evaluasi performa model (Laporan Subbab 2.1.10, 2.1.11, 3.5.2) – menjawab Rumusan Masalah 1.

Isi modul:
1. Confusion matrix (kelas positif = Malignant = 1).
2. Metrik evaluasi: accuracy, precision, recall, F1-score (Persamaan 2.9 – 2.12).
3. Stratified 10-Fold Cross Validation yang diulang 5 kali (50 pengujian per model).
4. Penentuan model terbaik: recall tertinggi; jika setara, dibandingkan F1-score.
5. Grafik confusion matrix dan perbandingan metrik.
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing.data import RANDOM_STATE, ImputerMedianManual, StandardScalerManual, stratified_kfold

METRIK = ["accuracy", "precision", "recall", "f1_score"]
NAMA_METRIK = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1_score": "F1-Score"}
HIJAU, KUNING, GELAP = "#3D6F0C", "#F9C22E", "#003D06"


# ---------------------------------------------------------------------------
# 1-2. Confusion matrix dan metrik
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _bagi(a, b):
    return a / b if b else 0.0


def hitung_metrik(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    accuracy = _bagi(tp + tn, tp + tn + fp + fn)   # (TP + TN) / total
    precision = _bagi(tp, tp + fp)                  # TP / (TP + FP)
    recall = _bagi(tp, tp + fn)                     # TP / (TP + FN)
    f1 = _bagi(2 * precision * recall, precision + recall)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, **cm}


# ---------------------------------------------------------------------------
# 3. Stratified 10-Fold Cross Validation x 5 (Subbab 3.5.2.1)
# ---------------------------------------------------------------------------
def validasi_kfold(daftar_model: dict, X, y, n_splits=10, n_repeats=5,
                   random_state=RANDOM_STATE, verbose=True) -> pd.DataFrame:
    """
    daftar_model: {"Random Forest": fungsi_pembuat_model, "SVM RBF": fungsi_pembuat_model}
    Setiap fold: imputer dan scaler di-fit pada data latih fold saja, lalu model dilatih dan diuji.
    Kedua model memakai pembagian fold yang sama agar perbandingannya adil.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    baris = []
    mulai = time.time()
    for ulangan, fold, tr, te in stratified_kfold(y, n_splits, n_repeats, random_state):
        # Preprocessing di dalam fold: imputer & scaler di-fit pada data latih fold saja
        imputer = ImputerMedianManual().fit(X[tr])
        scaler = StandardScalerManual().fit(imputer.transform(X[tr]))
        Xtr = scaler.transform(imputer.transform(X[tr]))
        Xte = scaler.transform(imputer.transform(X[te]))
        for nama, buat in daftar_model.items():
            model = buat().fit(Xtr, y[tr])
            m = hitung_metrik(y[te], model.predict(Xte))
            baris.append({"model": nama, "ulangan": ulangan, "fold": fold, **m})
        if verbose and fold == n_splits:
            print(f"  ulangan {ulangan}/{n_repeats} selesai ({time.time() - mulai:.0f} detik)")
    return pd.DataFrame(baris)


def ringkas_kfold(hasil: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata dan standar deviasi tiap metrik dari 50 pengujian."""
    agg = hasil.groupby("model")[METRIK].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def format_ringkasan(ringkasan: pd.DataFrame) -> pd.DataFrame:
    """Tabel siap tampil: 'rata-rata ± std' dalam persen."""
    out = pd.DataFrame({"Model": ringkasan["model"]})
    for m in METRIK:
        out[NAMA_METRIK[m]] = [f"{a * 100:.2f}% ± {s * 100:.2f}%"
                               for a, s in zip(ringkasan[f"{m}_mean"], ringkasan[f"{m}_std"])]
    return out


# ---------------------------------------------------------------------------
# 4. Penentuan model terbaik (Subbab 3.5.2)
# ---------------------------------------------------------------------------
def pilih_model_terbaik(ringkasan: pd.DataFrame, toleransi=1e-4) -> dict:
    r = ringkasan.set_index("model")
    urut = r.sort_values("recall_mean", ascending=False)
    a, b = urut.index[0], urut.index[1]
    selisih = urut.loc[a, "recall_mean"] - urut.loc[b, "recall_mean"]
    if selisih > toleransi:
        return {"model": a, "alasan": f"Recall rata-rata {a} lebih tinggi "
                                      f"({urut.loc[a, 'recall_mean']:.2%} vs {urut.loc[b, 'recall_mean']:.2%})."}
    terbaik = r["f1_score_mean"].idxmax()
    lain = b if terbaik == a else a
    return {"model": terbaik, "alasan": f"Recall kedua model setara, sehingga dibandingkan F1-score "
                                        f"({r.loc[terbaik, 'f1_score_mean']:.2%} vs {r.loc[lain, 'f1_score_mean']:.2%})."}


# ---------------------------------------------------------------------------
# 5. Grafik
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: dict, judul: str, path=None):
    """Baris = kelas aktual, kolom = hasil prediksi. FN ditandai kuning karena paling berbahaya."""
    nilai = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    label = [["TN", "FP"], ["FN", "TP"]]
    warna = [[HIJAU, "#E6EDDC"], [KUNING, HIJAU]]
    teks = [["white", GELAP], [GELAP, "white"]]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.97, 0.97, color=warna[i][j]))
            ax.text(j + 0.485, 1 - i + 0.56, str(nilai[i][j]), ha="center", va="center",
                    fontsize=22, fontweight="bold", color=teks[i][j])
            ax.text(j + 0.485, 1 - i + 0.26, label[i][j], ha="center", va="center", fontsize=10, color=teks[i][j])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Prediksi\nBenign", "Prediksi\nMalignant"])
    ax.set_yticks([1.485, 0.485], ["Aktual\nBenign", "Aktual\nMalignant"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(judul, fontsize=11, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_perbandingan_metrik(ringkasan: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lebar = 0.36
    x = np.arange(len(METRIK))
    for k, (warna, (_, r)) in enumerate(zip([HIJAU, KUNING], ringkasan.iterrows())):
        rata = [r[f"{m}_mean"] * 100 for m in METRIK]
        std = [r[f"{m}_std"] * 100 for m in METRIK]
        pos = x + (k - 0.5) * lebar
        ax.bar(pos, rata, lebar, yerr=std, capsize=4, color=warna, label=r["model"])
        for p, v in zip(pos, rata):
            ax.text(p, 81, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90,
                    color="white" if warna == HIJAU else GELAP, fontweight="bold")
    ax.set_xticks(x, [NAMA_METRIK[m] for m in METRIK])
    ax.set_ylabel("Nilai (%)")
    ax.set_ylim(80, 103)
    ax.set_title("Perbandingan Kinerja (Stratified 10-Fold CV × 5, rata-rata ± std)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
