"""
Generate dan simpan hasil Data Preparation ke folder preprocessing/hasil/.

Menjalankan alur yang sama dengan train.py (EDA -> rencana preprocessing ->
siapkan_data), lalu menyimpan setiap tahap sebagai file CSV agar hasil
preprocessing dapat diperiksa dan dilampirkan di laporan.

Cara menjalankan (dari folder utama proyek):
    python -m preprocessing.generate

File yang dihasilkan di preprocessing/hasil/:
    data_bersih.csv               30 fitur + label diagnosis (Malignant = 1, Benign = 0),
                                  kolom id sudah dihapus
    data_latih.csv                data latih 80% (nilai asli)
    data_uji.csv                  data uji 20% (nilai asli)
    data_latih_standardisasi.csv  data latih setelah z-score
    data_uji_standardisasi.csv    data uji setelah z-score
    parameter_standardisasi.csv   mu, sigma (dan median imputasi) per fitur dari data latih
    rencana_preprocessing.csv     temuan EDA -> tindakan -> alasan
    ringkasan_pembagian_data.csv  jumlah & proporsi kelas pada data latih/uji
    log_preprocessing.txt         log langkah preprocessing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import data as prep  # noqa: E402
from preprocessing import eda  # noqa: E402

FOLDER_HASIL = Path(__file__).resolve().parent / "hasil"


def _gabung(X, y, kolom_fitur) -> pd.DataFrame:
    """30 fitur + label diagnosis (Malignant = 1, Benign = 0); kolom id tidak disertakan."""
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=kolom_fitur)
    df["diagnosis"] = np.asarray(y, dtype=int)
    return df


def generate(folder: Path = FOLDER_HASIL) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)

    df = prep.muat_dataset()
    hasil_eda = eda.jalankan_eda(df)
    rencana = prep.rencana_preprocessing(hasil_eda)
    siap = prep.siapkan_data(df, hasil_eda)

    fitur = list(siap["X"].columns)

    tabel = {
        "data_bersih.csv": _gabung(siap["X"], siap["y"], fitur),
        "data_latih.csv": _gabung(siap["X_latih"], siap["y_latih"], fitur),
        "data_uji.csv": _gabung(siap["X_uji"], siap["y_uji"], fitur),
        "data_latih_standardisasi.csv": _gabung(siap["Xs_latih"], siap["y_latih"], fitur),
        "data_uji_standardisasi.csv": _gabung(siap["Xs_uji"], siap["y_uji"], fitur),
        "parameter_standardisasi.csv": pd.DataFrame({
            "fitur": fitur,
            "mean_latih (mu)": siap["scaler"].mean_,
            "std_latih (sigma)": siap["scaler"].scale_,
            "median_imputasi": siap["imputer"].median_,
        }),
        "rencana_preprocessing.csv": pd.DataFrame(rencana),
        "ringkasan_pembagian_data.csv": pd.DataFrame([
            {"bagian": nama, "jumlah": len(y), "malignant": int(y.sum()), "benign": int((y == 0).sum()),
             "persen_malignant": round(float(y.mean()) * 100, 1)}
            for nama, y in (("Data latih", siap["y_latih"]), ("Data uji", siap["y_uji"]),
                            ("Total", siap["y"]))
        ]),
    }

    paths = {}
    for nama, isi in tabel.items():
        path = folder / nama
        isi.to_csv(path, index=False)
        paths[nama] = path

    log_path = folder / "log_preprocessing.txt"
    log_path.write_text("\n".join(f"- {baris}" for baris in siap["log"]) + "\n", encoding="utf-8")
    paths["log_preprocessing.txt"] = log_path
    return paths


if __name__ == "__main__":
    hasil = generate()
    print(f"Hasil preprocessing disimpan di {FOLDER_HASIL.relative_to(ROOT)}/")
    for nama, path in hasil.items():
        print(f"  - {nama}")
