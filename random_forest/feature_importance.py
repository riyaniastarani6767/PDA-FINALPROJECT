"""
Interpretasi model Random Forest menggunakan Feature Importance berbasis
penurunan Entropy / Information Gain

Cara perhitungan:
1. Pada setiap pohon keputusan di dalam Random Forest, setiap simpul internal (split node)
   memisahkan data menggunakan suatu fitur A dan menghasilkan nilai Information Gain IG(S, A)
2. Kontribusi fitur A pada simpul tersebut dihitung sebesar penurunan ketidakmurnian
   (impurity decrease) tertimbang proporsi sampel yang melewati simpul:
       delta_I(S, A) = (|S| / |S_root|) * IG(S, A)
3. Nilai kontribusi dijumlahkan untuk setiap fitur di dalam satu pohon lalu dinormalisasi,
   kemudian dirata-ratakan di seluruh n_estimators pohon pada Random Forest sehingga
   total nilai importance seluruh 30 fitur berjumlah 1,0 (100%).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.data import FITUR, label_fitur

HIJAU = "#3D6F0C"


def _akumulasi_pohon(node: dict, n_total: int, skor: np.ndarray) -> None:
    """Menelusuri simpul internal pohon secara rekursif untuk menjumlahkan penurunan entropy."""
    if node.get("leaf", True):
        return
    proporsi = node["n_samples"] / n_total if n_total > 0 else 0.0
    skor[node["fitur"]] += proporsi * node["information_gain"]
    _akumulasi_pohon(node["left"], n_total, skor)
    _akumulasi_pohon(node["right"], n_total, skor)


def hitung_feature_importance(model_rf) -> np.ndarray:
    """
    Menghitung vektor feature importance (berukuran jumlah fitur) dari seluruh pohon
    pada model RandomForestManual.
    """
    n_fitur = getattr(model_rf, "n_features_", len(FITUR))
    if not getattr(model_rf, "trees", None):
        return np.zeros(n_fitur, dtype=float)

    total = np.zeros(n_fitur, dtype=float)
    for pohon in model_rf.trees:
        skor_pohon = np.zeros(n_fitur, dtype=float)
        n_root = pohon.get("n_samples", 1)
        _akumulasi_pohon(pohon, n_root, skor_pohon)
        jumlah = skor_pohon.sum()
        if jumlah > 0:
            total += skor_pohon / jumlah

    if total.sum() > 0:
        total = total / total.sum()
    return total


def tabel_feature_importance(model_rf, fitur: list[str] = FITUR) -> pd.DataFrame:
    """
    Menyusun tabel urutan fitur paling berpengaruh pada Random Forest
    berdasarkan nilai feature importance tertinggi ke terendah.

    Kolom yang dihasilkan:
    - peringkat  : 1 .. 30
    - fitur      : nama kolom fitur pada dataset (mis. 'concave_points_worst')
    - nama       : nama fitur yang mudah dibaca (mis. 'Concave points (worst)')
    - importance : proporsi kontribusi fitur (total seluruh fitur = 1.0)
    """
    skor = hitung_feature_importance(model_rf)
    df = pd.DataFrame({
        "fitur": list(fitur),
        "nama": [label_fitur(f) for f in fitur],
        "importance": np.asarray(skor, dtype=float),
    })
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.insert(0, "peringkat", range(1, len(df) + 1))
    return df


def plot_feature_importance(tabel_fi: pd.DataFrame, top: int = 10, path=None):
    """
    Membuat grafik batang horizontal (horizontal bar chart) untuk menampilkan
    fitur-fitur dengan nilai feature importance tertinggi pada Random Forest.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = tabel_fi.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.45 * top + 1.2))
    ax.barh(data["nama"], data["importance"], color=HIJAU)
    maks = float(data["importance"].max()) if len(data) else 0.1
    ax.set_xlim(0, maks * 1.15)
    for i, v in enumerate(data["importance"]):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Nilai Feature Importance (Penurunan Entropy / Information Gain)")
    ax.set_title(f"{top} Fitur Paling Berpengaruh – Random Forest (Feature Importance)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
