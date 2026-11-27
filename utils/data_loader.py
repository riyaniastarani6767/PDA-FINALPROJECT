from pathlib import Path

import pandas as pd

from preprocessing.data import FITUR, label_fitur


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "breast_cancer.csv"


class DatasetLoadError(RuntimeError):
    """Kesalahan ketika dataset atau struktur kolom proyek tidak dapat dibaca."""


def muat_dataset_ui() -> pd.DataFrame:
    """Memuat CSV asli dan memastikan 30 fitur dari preprocessing/data.py tersedia."""
    if not DATASET_PATH.exists():
        raise DatasetLoadError(f"File dataset tidak ditemukan: {DATASET_PATH}")

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as exc:
        raise DatasetLoadError(f"Dataset gagal dibaca: {exc}") from exc

    kolom_kurang = [nama for nama in FITUR if nama not in df.columns]
    if "diagnosis" not in df.columns:
        kolom_kurang.append("diagnosis")
    if kolom_kurang:
        raise DatasetLoadError("Kolom wajib tidak ditemukan: " + ", ".join(kolom_kurang))

    if df.empty:
        raise DatasetLoadError("Dataset tersedia tetapi tidak memiliki baris data.")

    return df


def ringkasan_dataset(df: pd.DataFrame) -> dict:
    """Menghasilkan ringkasan murni dari CSV, tanpa nilai dummy."""
    kelas = df["diagnosis"].astype(str).str.upper().value_counts()
    benign = int(kelas.get("B", 0))
    malignant = int(kelas.get("M", 0))
    total = benign + malignant

    return {
        "total_sampel": int(len(df)),
        "total_fitur": len(FITUR),
        "benign": benign,
        "malignant": malignant,
        "persen_benign": round(benign / total * 100, 1) if total else 0.0,
        "persen_malignant": round(malignant / total * 100, 1) if total else 0.0,
        "jumlah_kolom": int(df.shape[1]),
    }


def tabel_daftar_fitur() -> pd.DataFrame:
    """Menyusun 30 nama fitur asli berdasarkan FITUR dari preprocessing/data.py."""
    baris = []
    for nama in FITUR:
        if nama.endswith("_mean"):
            kategori = "Mean"
        elif nama.endswith("_se"):
            kategori = "Standard Error"
        elif nama.endswith("_worst"):
            kategori = "Worst"
        else:
            kategori = "Lainnya"
        baris.append({
            "No": len(baris) + 1,
            "Nama fitur asli": nama,
            "Nama tampilan": label_fitur(nama),
            "Kategori": kategori,
        })
    return pd.DataFrame(baris)


def statistik_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik deskriptif untuk 30 fitur numerik asli."""
    numerik = df[FITUR].apply(pd.to_numeric, errors="coerce")
    stat = pd.DataFrame({
        "Fitur": [label_fitur(nama) for nama in FITUR],
        "Nama asli": FITUR,
        "Mean": numerik.mean().to_numpy(),
        "Median": numerik.median().to_numpy(),
        "Standard deviation": numerik.std(ddof=1).to_numpy(),
        "Minimum": numerik.min().to_numpy(),
        "Maximum": numerik.max().to_numpy(),
    })
    return stat


def status_artefak_model() -> dict[str, str]:
    """Memeriksa keberadaan file; keberadaan file tidak berarti model terverifikasi."""
    model_dir = ROOT / "models"
    rf = model_dir / "model_rf.pkl"
    svm = model_dir / "model_svm.pkl"

    return {
        "Random Forest": "Tersedia—belum terverifikasi" if rf.exists() else "Belum tersedia",
        "Support Vector Machine": "Tersedia—belum terverifikasi" if svm.exists() else "Belum tersedia",
    }
