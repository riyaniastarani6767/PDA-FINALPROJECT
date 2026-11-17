"""
Tahap CRISP-DM: Data Understanding dan Data Preparation
(Laporan Subbab 3.1, 3.2.2, dan 3.2.3).

Isi modul:
1. Memuat dataset Breast Cancer Wisconsin (Diagnostic).
2. Data Preparation yang DITENTUKAN dari hasil EDA (preprocessing/eda.py):
   setiap langkah dicatat beserta temuan EDA yang menjadi alasannya.
   a. Duplikat        -> dihapus jika ada
   b. Missing value   -> imputasi median dari data latih jika ada
   c. Outlier         -> dideteksi (IQR), diputuskan tetap dipertahankan
   d. Kolom ID        -> dihapus (bukan informasi diagnostik)
   e. Label           -> Malignant = 1, Benign = 0
   f. Korelasi tinggi -> 30 fitur tetap dipakai (tidak ada reduksi)
   g. Imbalance       -> pembagian 80 : 20 secara stratified (manual)
   h. Perbedaan skala -> standardisasi z-score (mu, sigma dari data latih)
3. Stratified K-Fold manual (dipakai evaluasi dan Platt scaling SVM).
4. Validasi data masukan dari aplikasi (kebutuhan fungsional F-02, F-04, F-05).
"""
from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# 10 karakteristik inti sel x 3 variasi (mean, se, worst) = 30 fitur.
KARAKTERISTIK = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
VARIASI = ["mean", "se", "worst"]
# Urutan kolom sesuai file wdbc.data: 10 mean, 10 se, 10 worst.
FITUR = [f"{k}_{v}" for v in VARIASI for k in KARAKTERISTIK]

NAMA_KARAKTERISTIK = {
    "radius": "Radius", "texture": "Texture", "perimeter": "Perimeter", "area": "Area",
    "smoothness": "Smoothness", "compactness": "Compactness", "concavity": "Concavity",
    "concave_points": "Concave points", "symmetry": "Symmetry",
    "fractal_dimension": "Fractal dimension",
}
LABEL_KELAS = {1: "Malignant", 0: "Benign"}
RANDOM_STATE = 42


def label_fitur(nama: str) -> str:
    """Nama fitur yang mudah dibaca, contoh: 'radius_mean' -> 'Radius (mean)'."""
    k, v = nama.rsplit("_", 1)
    return f"{NAMA_KARAKTERISTIK.get(k, k)} ({v.upper() if v == 'se' else v})"


# ---------------------------------------------------------------------------
# 1. Memuat dataset (Data Understanding)
# ---------------------------------------------------------------------------
def _rapikan_kolom(df: pd.DataFrame) -> pd.DataFrame:
    """Menyeragamkan nama kolom, misalnya 'concave points_mean' -> 'concave_points_mean'."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df.loc[:, ~df.columns.str.startswith("unnamed")]


def muat_dataset() -> pd.DataFrame:
    """
    Memuat dataset dengan urutan prioritas:
    1. data/wdbc.data (file asli UCI, tanpa header)
    2. file CSV di folder data/ yang memiliki kolom 'diagnosis'
    3. salinan dataset yang sama dari scikit-learn (lalu disimpan ke data/breast_cancer.csv)

    Hasil: DataFrame berisi kolom id, diagnosis (M/B), dan 30 fitur.
    """
    wdbc = DATA_DIR / "wdbc.data"
    if wdbc.exists():
        return pd.read_csv(wdbc, header=None, names=["id", "diagnosis"] + FITUR)

    for csv in sorted(DATA_DIR.glob("*.csv")):
        df = _rapikan_kolom(pd.read_csv(csv))
        if "diagnosis" in df.columns and set(FITUR).issubset(df.columns):
            if "id" not in df.columns:
                df.insert(0, "id", range(1, len(df) + 1))
            return df[["id", "diagnosis"] + FITUR]

    # Cadangan terakhir jika folder data/ kosong: salinan dataset yang sama dari scikit-learn.
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=FITUR)
    # Pada scikit-learn: target 0 = malignant, 1 = benign.
    df.insert(0, "diagnosis", np.where(raw.target == 0, "M", "B"))
    df.insert(0, "id", range(1, len(df) + 1))
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "breast_cancer.csv", index=False)
    return df


def cek_kualitas(df: pd.DataFrame) -> dict:
    """Ringkasan kualitas data untuk tahap Data Understanding (Subbab 3.2.2)."""
    dist = df["diagnosis"].value_counts()
    return {
        "jumlah_data": int(len(df)),
        "jumlah_fitur": len(FITUR),
        "missing_value": int(df[FITUR + ["diagnosis"]].isna().sum().sum()),
        "duplikat": int(df.drop(columns="id").duplicated().sum()),
        "benign": int(dist.get("B", 0)),
        "malignant": int(dist.get("M", 0)),
        "persen_benign": round(float(dist.get("B", 0)) / len(df) * 100, 1),
        "persen_malignant": round(float(dist.get("M", 0)) / len(df) * 100, 1),
    }


# ---------------------------------------------------------------------------
# 2. Data Preparation berdasarkan hasil EDA
# ---------------------------------------------------------------------------
def bersihkan(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Menghapus kolom ID dan mengubah label target: Malignant = 1, Benign = 0."""
    X = df[FITUR].astype(float).reset_index(drop=True)
    y = (df["diagnosis"].astype(str).str.upper() == "M").astype(int).to_numpy()
    return X, y


def rencana_preprocessing(hasil_eda: dict) -> list[dict]:
    """
    Menyusun langkah Data Preparation dari temuan EDA.
    Setiap langkah berisi: temuan EDA, tindakan, dan alasan.
    """
    q, d, sk, o = hasil_eda["kualitas"], hasil_eda["distribusi_kelas"], hasil_eda["skala"], hasil_eda["outlier"]
    kt = hasil_eda["pasangan_korelasi_tinggi"]
    return [
        {"langkah": "Penanganan duplikat",
         "temuan": f"{q['duplikat']} baris duplikat",
         "tindakan": "Hapus baris duplikat" if q["duplikat"] else "Tidak ada yang dihapus",
         "alasan": "Data ganda membuat model belajar dari pola yang sama dua kali dan dapat bocor ke data uji."},
        {"langkah": "Penanganan missing value",
         "temuan": f"{q['missing_value']} nilai kosong",
         "tindakan": "Imputasi median dari data latih" if q["missing_value"] else "Tidak perlu imputasi",
         "alasan": "Model tidak dapat memproses nilai kosong; median dipilih karena tahan terhadap outlier."},
        {"langkah": "Penanganan outlier",
         "temuan": f"{o['pasien_dengan_outlier']} pasien memiliki outlier (IQR); "
                   f"{o['persen_nilai_outlier_dari_malignant']}% nilai outlier berasal dari malignant",
         "tindakan": "Outlier dipertahankan (tidak dihapus)",
         "alasan": "Nilai ekstrem didominasi kelas malignant, yaitu ciri inti sel ganas yang lebih besar dan tidak "
                   "beraturan; menghapusnya membuang informasi diagnostik. Pengaruh skala ditangani z-score."},
        {"langkah": "Penghapusan kolom ID",
         "temuan": "Kolom id hanya nomor identitas pasien",
         "tindakan": "Hapus kolom id",
         "alasan": "Tidak menyimpan informasi diagnostik sehingga tidak boleh dipelajari model."},
        {"langkah": "Transformasi label",
         "temuan": f"Label berupa huruf {', '.join(q['label_unik'])}",
         "tindakan": "Malignant = 1, Benign = 0",
         "alasan": "Model membutuhkan label numerik; malignant ditetapkan sebagai kelas positif."},
        {"langkah": "Seleksi fitur",
         "temuan": f"{kt['jumlah']} pasangan fitur berkorelasi > {kt['ambang']} (misalnya radius, perimeter, area)",
         "tindakan": "Seluruh 30 fitur tetap digunakan",
         "alasan": "Random Forest memilih fitur secara acak di setiap node, SVM RBF memakai semua fitur dalam "
                   "perhitungan jarak, dan SHAP perlu menilai kontribusi setiap fitur (Rumusan Masalah 2)."},
        {"langkah": "Pembagian data",
         "temuan": f"Kelas tidak seimbang: {d['persen_benign']}% benign, {d['persen_malignant']}% malignant "
                   f"(rasio {d['rasio_mayoritas_minoritas']} : 1)",
         "tindakan": "Hold-out 80 : 20 secara stratified",
         "alasan": "Menjaga proporsi kelas tetap sama pada data latih dan data uji."},
        {"langkah": "Standardisasi fitur",
         "temuan": f"Skala berbeda jauh: {sk['fitur_nilai_terbesar']} sampai {sk['nilai_maks_terbesar']}, "
                   f"{sk['fitur_nilai_terkecil']} maksimum {sk['nilai_maks_terkecil']}",
         "tindakan": "z-score dengan mu dan sigma dari data latih",
         "alasan": "Kernel RBF berbasis jarak sehingga fitur berskala besar akan mendominasi (Subbab 2.1.8.1)."},
    ]


class ImputerMedianManual:
    """Mengisi nilai kosong dengan median setiap fitur yang dihitung dari data latih."""

    def fit(self, X):
        self.median_ = np.nanmedian(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        baris, kolom = np.where(np.isnan(X))
        X[baris, kolom] = self.median_[kolom]
        return X


class StandardScalerManual:
    """
    Standardisasi z-score: z = (x - mu) / sigma.
    mu dan sigma dihitung HANYA dari data latih, lalu dipakai ulang untuk
    data uji dan data pasien baru (mencegah data leakage).
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _jumlah_uji_per_kelas(n_kelas: dict, n_uji: int) -> dict:
    """Membagi jumlah data uji ke setiap kelas secara proporsional (sisa terbesar)."""
    total = sum(n_kelas.values())
    kuota = {k: n * n_uji / total for k, n in n_kelas.items()}
    hasil = {k: int(math.floor(v)) for k, v in kuota.items()}
    sisa = n_uji - sum(hasil.values())
    for k in sorted(kuota, key=lambda k: kuota[k] - hasil[k], reverse=True)[:sisa]:
        hasil[k] += 1
    return hasil


def bagi_stratified(X, y, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Pembagian hold-out stratified secara manual (569 data -> 455 latih, 114 uji).
    Data setiap kelas diacak, lalu diambil sejumlah proporsional sebagai data uji.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    kelas = np.unique(y)
    n_uji = int(math.ceil(test_size * len(y)))
    jatah = _jumlah_uji_per_kelas({k: int((y == k).sum()) for k in kelas}, n_uji)
    idx_uji = []
    for k in kelas:
        idx = rng.permutation(np.flatnonzero(y == k))
        idx_uji.extend(idx[:jatah[k]])
    idx_uji = np.sort(np.array(idx_uji))
    idx_latih = np.setdiff1d(np.arange(len(y)), idx_uji)
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx_latih], X.iloc[idx_uji], y[idx_latih], y[idx_uji]
    X = np.asarray(X)
    return X[idx_latih], X[idx_uji], y[idx_latih], y[idx_uji]


# Nama lama tetap tersedia
bagi_data = bagi_stratified


def stratified_kfold(y, n_splits: int = 10, n_repeats: int = 1, random_state: int = RANDOM_STATE):
    """
    Stratified K-Fold manual (Subbab 2.1.6 dan 3.5.2.1).
    Data setiap kelas diacak lalu dibagi rata ke k fold, sehingga proporsi kelas
    di setiap fold mendekati data asli. Diulang n_repeats kali dengan pengacakan berbeda.
    Menghasilkan (ulangan, fold, indeks_latih, indeks_uji).
    """
    y = np.asarray(y)
    semua = np.arange(len(y))
    for ulang in range(n_repeats):
        rng = np.random.default_rng(random_state + ulang)
        fold = [[] for _ in range(n_splits)]
        geser = 0
        for k in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == k))
            bagian = np.array_split(idx, n_splits)
            # geser urutan agar ukuran fold tetap seimbang antar kelas
            for i, b in enumerate(bagian):
                fold[(i + geser) % n_splits].extend(b)
            geser += len(idx) % n_splits
        for i, uji in enumerate(fold):
            uji = np.sort(np.array(uji, dtype=int))
            yield ulang + 1, i + 1, np.setdiff1d(semua, uji), uji


def siapkan_data(df: pd.DataFrame, hasil_eda: dict, test_size: float = 0.2,
                 random_state: int = RANDOM_STATE) -> dict:
    """
    Menjalankan Data Preparation sesuai rencana dari EDA.
    Mengembalikan data latih/uji (asli dan terstandardisasi), scaler, imputer, dan log langkah.
    """
    log = []
    q = hasil_eda["kualitas"]

    data = df.copy()
    if q["duplikat"] > 0:
        sebelum = len(data)
        data = data.loc[~data.drop(columns="id").duplicated()].reset_index(drop=True)
        log.append(f"Hapus duplikat: {sebelum - len(data)} baris dihapus")
    else:
        log.append("Duplikat: tidak ada, tidak ada baris yang dihapus")

    log.append(f"Outlier: {hasil_eda['outlier']['pasien_dengan_outlier']} pasien memiliki outlier, dipertahankan")

    X, y = bersihkan(data)
    log.append("Kolom id dihapus; label diubah: Malignant = 1, Benign = 0")
    log.append(f"Seleksi fitur: tidak dilakukan, {X.shape[1]} fitur digunakan")

    X_latih, X_uji, y_latih, y_uji = bagi_stratified(X, y, test_size, random_state)
    log.append(f"Split stratified: {len(y_latih)} latih ({y_latih.mean():.1%} malignant), "
               f"{len(y_uji)} uji ({y_uji.mean():.1%} malignant)")

    imputer = ImputerMedianManual().fit(X_latih)
    if q["missing_value"] > 0:
        log.append(f"Missing value: {q['missing_value']} nilai diisi median data latih")
    else:
        log.append("Missing value: tidak ada, tidak perlu imputasi")

    scaler = StandardScalerManual().fit(imputer.transform(X_latih))
    log.append("Standardisasi z-score: mu dan sigma dihitung dari data latih")

    return {
        "X": X, "y": y,
        "X_latih": X_latih, "X_uji": X_uji, "y_latih": y_latih, "y_uji": y_uji,
        "Xs_latih": scaler.transform(imputer.transform(X_latih)),
        "Xs_uji": scaler.transform(imputer.transform(X_uji)),
        "imputer": imputer, "scaler": scaler, "log": log,
    }


# ---------------------------------------------------------------------------
# 3. Validasi data masukan aplikasi (F-02, F-04, F-05)
# ---------------------------------------------------------------------------
def validasi_satu_pasien(nilai: dict) -> list[str]:
    """Memeriksa 30 fitur input manual. Mengembalikan daftar pesan kesalahan (kosong = valid)."""
    kosong = [label_fitur(f) for f in FITUR if nilai.get(f) is None or pd.isna(nilai.get(f))]
    if kosong:
        return [f"Data belum lengkap: {len(kosong)} fitur masih kosong ({', '.join(kosong[:5])}"
                f"{', ...' if len(kosong) > 5 else ''})."]
    negatif = [label_fitur(f) for f in FITUR if float(nilai[f]) < 0]
    if negatif:
        return [f"Nilai fitur tidak boleh negatif: {', '.join(negatif)}."]
    return []


def validasi_csv(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Memeriksa file CSV yang diunggah:
    - semua 30 kolom fitur tersedia,
    - nilainya berupa angka,
    - tidak ada nilai kosong,
    - tidak ada nilai negatif.
    Mengembalikan (DataFrame 30 fitur yang sudah rapi, daftar pesan kesalahan).
    """
    if df is None or df.empty:
        return None, ["File CSV kosong."]
    df = _rapikan_kolom(df)
    kurang = [f for f in FITUR if f not in df.columns]
    if kurang:
        return None, [f"Kolom CSV tidak sesuai: {len(kurang)} kolom fitur tidak ditemukan "
                      f"({', '.join(kurang[:5])}{', ...' if len(kurang) > 5 else ''})."]
    data = df[FITUR].apply(pd.to_numeric, errors="coerce")
    salah = [f for f in FITUR if data[f].isna().any() and df[f].notna().all()]
    if salah:
        return None, [f"Format tidak valid: kolom {', '.join(salah[:5])} berisi nilai yang bukan angka."]
    if data.isna().any().any():
        baris = (data.isna().any(axis=1).to_numpy().nonzero()[0] + 1).tolist()
        return None, [f"Terdapat nilai kosong pada baris ke-{', '.join(map(str, baris[:10]))}."]
    if (data < 0).any().any():
        return None, ["Terdapat nilai fitur negatif. Nilai karakteristik inti sel tidak boleh negatif."]
    return data.astype(float).reset_index(drop=True), []
