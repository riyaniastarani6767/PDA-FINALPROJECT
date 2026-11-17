"""
Exploratory Data Analysis (EDA) – fase CRISP-DM Data Understanding
(Laporan Subbab 2.1.4 dan 3.2.2).

EDA menjawab lima pertanyaan (slide teori EDA):
1. Adakah missing value, duplikat, atau outlier?          -> kualitas data, outlier IQR
2. Apakah kelas target tidak seimbang?                     -> distribusi kelas
3. Bagaimana distribusi setiap atribut?                    -> univariat (statistik, skala, histogram)
4. Atribut mana yang punya daya pemisah terhadap target?   -> bivariat (per kelas, korelasi target)
5. Bagaimana hubungan antarvariabel?                       -> multivariat (heatmap korelasi)

Hasil EDA (dict) dipakai oleh preprocessing/data.py untuk MENENTUKAN langkah
Data Preparation, sehingga setiap langkah preprocessing punya dasar dari data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.data import FITUR, KARAKTERISTIK, label_fitur

HIJAU, KUNING, GELAP = "#3D6F0C", "#F9C22E", "#003D06"
WARNA_KELAS = {"B": HIJAU, "M": KUNING}
FITUR_MEAN = [f"{k}_mean" for k in KARAKTERISTIK]
AMBANG_KORELASI = 0.9


# ---------------------------------------------------------------------------
# 1. Kualitas data
# ---------------------------------------------------------------------------
def kualitas_data(df: pd.DataFrame) -> dict:
    fitur = df[FITUR]
    missing_per_kolom = df[["diagnosis"] + FITUR].isna().sum()
    return {
        "jumlah_data": int(len(df)),
        "jumlah_kolom": int(df.shape[1]),
        "jumlah_fitur": len(FITUR),
        "fitur_numerik": int(fitur.select_dtypes("number").shape[1]),
        "missing_value": int(missing_per_kolom.sum()),
        "kolom_missing": {k: int(v) for k, v in missing_per_kolom.items() if v > 0},
        "duplikat": int(df.drop(columns="id").duplicated().sum()),
        "id_duplikat": int(df["id"].duplicated().sum()),
        "nilai_negatif": int((fitur < 0).sum().sum()),
        "label_unik": sorted(df["diagnosis"].dropna().astype(str).unique().tolist()),
    }


# ---------------------------------------------------------------------------
# 2. Distribusi kelas (imbalance)
# ---------------------------------------------------------------------------
def distribusi_kelas(df: pd.DataFrame) -> dict:
    n = df["diagnosis"].value_counts()
    b, m = int(n.get("B", 0)), int(n.get("M", 0))
    return {
        "benign": b, "malignant": m,
        "persen_benign": round(b / len(df) * 100, 1),
        "persen_malignant": round(m / len(df) * 100, 1),
        "rasio_mayoritas_minoritas": round(max(b, m) / min(b, m), 2),
        "tidak_seimbang": bool(abs(b - m) / len(df) > 0.1),
    }


# ---------------------------------------------------------------------------
# 3. Univariat: statistik deskriptif dan perbedaan skala
# ---------------------------------------------------------------------------
def statistik_deskriptif(df: pd.DataFrame) -> pd.DataFrame:
    s = df[FITUR].agg(["min", "max", "mean", "median", "std"]).T
    s["rentang"] = s["max"] - s["min"]
    return s


def perbedaan_skala(stat: pd.DataFrame) -> dict:
    besar, kecil = stat["max"].idxmax(), stat["max"].idxmin()
    return {
        "fitur_nilai_terbesar": besar, "nilai_maks_terbesar": round(float(stat.loc[besar, "max"]), 4),
        "fitur_nilai_terkecil": kecil, "nilai_maks_terkecil": round(float(stat.loc[kecil, "max"]), 4),
        "rasio_rentang": round(float(stat["rentang"].max() / stat["rentang"].min()), 1),
        "perlu_penskalaan": bool(stat["rentang"].max() / stat["rentang"].min() > 10),
    }


# ---------------------------------------------------------------------------
# 4. Outlier (metode IQR)
# ---------------------------------------------------------------------------
def deteksi_outlier(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Outlier jika nilai < Q1 - 1,5 IQR atau > Q3 + 1,5 IQR."""
    q1, q3 = df[FITUR].quantile(0.25), df[FITUR].quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[FITUR] < bawah) | (df[FITUR] > atas)
    is_m = df["diagnosis"] == "M"
    tabel = pd.DataFrame({
        "batas_bawah": bawah, "batas_atas": atas,
        "jumlah_outlier": mask.sum(),
        "persen": (mask.mean() * 100).round(1),
        "dari_malignant": mask[is_m].sum(),
        "dari_benign": mask[~is_m].sum(),
    }).sort_values("jumlah_outlier", ascending=False)
    baris = mask.any(axis=1)
    total = int(mask.sum().sum())
    ringkas = {
        "total_nilai_outlier": total,
        "total_nilai": int(mask.size),
        "pasien_dengan_outlier": int(baris.sum()),
        "pasien_outlier_malignant": int(baris[is_m].sum()),
        "pasien_outlier_benign": int(baris[~is_m].sum()),
        "persen_malignant_punya_outlier": round(float(baris[is_m].mean() * 100), 1),
        "persen_benign_punya_outlier": round(float(baris[~is_m].mean() * 100), 1),
        "persen_nilai_outlier_dari_malignant": round(float(mask[is_m].sum().sum()) / total * 100, 1) if total else 0.0,
        "di_atas_batas_atas": int((df[FITUR] > atas).sum().sum()),
        "di_bawah_batas_bawah": int((df[FITUR] < bawah).sum().sum()),
    }
    return tabel, ringkas


# ---------------------------------------------------------------------------
# 5. Bivariat & multivariat: korelasi
# ---------------------------------------------------------------------------
def korelasi_target(df: pd.DataFrame) -> pd.Series:
    """Korelasi Pearson setiap fitur dengan diagnosis (M = 1, B = 0)."""
    y = (df["diagnosis"] == "M").astype(int)
    return df[FITUR].corrwith(y).sort_values(ascending=False)


def rasio_rata_kelas(df: pd.DataFrame) -> pd.DataFrame:
    """Perbandingan rata-rata malignant terhadap benign (daya pemisah)."""
    r = df.groupby("diagnosis")[FITUR].mean().T
    r.columns = [f"rata_{c}" for c in r.columns]
    r["rasio_M_per_B"] = r["rata_M"] / r["rata_B"]
    return r.sort_values("rasio_M_per_B", ascending=False)


def pasangan_korelasi_tinggi(df: pd.DataFrame, ambang=AMBANG_KORELASI) -> pd.DataFrame:
    c = df[FITUR].corr().abs()
    atas = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack()
    p = atas[atas > ambang].sort_values(ascending=False).reset_index()
    p.columns = ["fitur_1", "fitur_2", "korelasi"]
    return p


# ---------------------------------------------------------------------------
# Grafik
# ---------------------------------------------------------------------------
def _simpan(fig, folder, nama):
    if folder is not None:
        fig.savefig(Path(folder) / nama, dpi=130, bbox_inches="tight")
    return fig


def plot_distribusi_kelas(df, folder=None):
    d = distribusi_kelas(df)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    bar = ax.bar(["Benign (B)", "Malignant (M)"], [d["benign"], d["malignant"]], color=[HIJAU, KUNING])
    for b, v, p in zip(bar, [d["benign"], d["malignant"]], [d["persen_benign"], d["persen_malignant"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v} ({p}%)", ha="center")
    ax.set_ylim(0, max(d["benign"], d["malignant"]) * 1.18)
    ax.set_title(f"Distribusi kelas (rasio {d['rasio_mayoritas_minoritas']} : 1)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _simpan(fig, folder, "eda_distribusi_kelas.png")


def plot_skala_fitur(stat, folder=None):
    s = stat["max"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh([label_fitur(f) for f in s.index], s.values, color=[KUNING if v > 100 else HIJAU for v in s.values])
    ax.set_xscale("log")
    ax.set_xlabel("Nilai maksimum (skala log)")
    ax.set_title("Perbedaan skala 30 fitur")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _simpan(fig, folder, "eda_skala_fitur.png")


def plot_histogram(df, folder=None):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, f in zip(axes.ravel(), FITUR_MEAN):
        for kelas in ["B", "M"]:
            ax.hist(df.loc[df["diagnosis"] == kelas, f], bins=30, alpha=0.6, color=WARNA_KELAS[kelas],
                    label="Benign" if kelas == "B" else "Malignant")
        ax.set_title(label_fitur(f), fontsize=10)
    axes[0, 0].legend()
    fig.suptitle("Distribusi 10 fitur mean per kelas (univariat & bivariat)", fontsize=14)
    fig.tight_layout()
    return _simpan(fig, folder, "eda_histogram_mean.png")


def plot_boxplot_outlier(df, folder=None):
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, f in zip(axes.ravel(), FITUR_MEAN):
        data = [df.loc[df["diagnosis"] == k, f] for k in ["B", "M"]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#C0392B", markeredgecolor="#C0392B"))
        for patch, k in zip(bp["boxes"], ["B", "M"]):
            patch.set_facecolor(WARNA_KELAS[k])
        ax.set_xticks([1, 2], ["Benign", "Malignant"])
        ax.set_title(label_fitur(f), fontsize=10)
    fig.suptitle("Boxplot 10 fitur mean per kelas – titik merah = outlier (IQR)", fontsize=14)
    fig.tight_layout()
    return _simpan(fig, folder, "eda_boxplot_outlier.png")


def plot_korelasi_target(df, folder=None):
    k = korelasi_target(df).sort_values()
    fig, ax = plt.subplots(figsize=(8, 8.5))
    ax.barh([label_fitur(f) for f in k.index], k.values, color=[KUNING if v > 0.5 else HIJAU for v in k.values])
    ax.axvline(0.5, ls="--", color="#999", lw=0.8)
    ax.set_xlabel("Korelasi dengan diagnosis (M = 1)")
    ax.set_title("Daya pemisah fitur terhadap diagnosis (bivariat)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _simpan(fig, folder, "eda_korelasi_target.png")


def plot_heatmap(df, folder=None):
    c = df[FITUR_MEAN].corr()
    nama = [label_fitur(f).replace(" (mean)", "") for f in FITUR_MEAN]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(c, cmap="YlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(10), nama, rotation=45, ha="right")
    ax.set_yticks(range(10), nama)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if c.iloc[i, j] > 0.7 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Korelasi antar 10 fitur mean (multivariat)")
    fig.tight_layout()
    return _simpan(fig, folder, "eda_heatmap_korelasi.png")


# ---------------------------------------------------------------------------
# Menjalankan seluruh EDA
# ---------------------------------------------------------------------------
def jalankan_eda(df: pd.DataFrame, folder_gambar=None) -> dict:
    """Menjalankan seluruh EDA dan mengembalikan ringkasan temuan (dapat disimpan ke JSON)."""
    stat = statistik_deskriptif(df)
    tabel_outlier, outlier = deteksi_outlier(df)
    kor = korelasi_target(df)
    tinggi = pasangan_korelasi_tinggi(df)
    hasil = {
        "kualitas": kualitas_data(df),
        "distribusi_kelas": distribusi_kelas(df),
        "skala": perbedaan_skala(stat),
        "outlier": outlier,
        "outlier_per_fitur": tabel_outlier.head(10)[["jumlah_outlier", "persen", "dari_malignant", "dari_benign"]]
            .astype(float).round(1).to_dict(orient="index"),
        "korelasi_target_teratas": kor.head(5).round(3).to_dict(),
        "korelasi_target_terendah": kor.tail(5).round(3).to_dict(),
        "pasangan_korelasi_tinggi": {
            "ambang": AMBANG_KORELASI,
            "jumlah": int(len(tinggi)),
            "contoh": tinggi.head(5).round(3).to_dict(orient="records"),
        },
    }
    if folder_gambar is not None:
        Path(folder_gambar).mkdir(parents=True, exist_ok=True)
        for fungsi in (plot_distribusi_kelas, plot_histogram, plot_boxplot_outlier,
                       plot_korelasi_target, plot_heatmap):
            plt.close(fungsi(df, folder_gambar))
        plt.close(plot_skala_fitur(stat, folder_gambar))
    return hasil
