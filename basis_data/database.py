"""
Basis data SQLite sistem klasifikasi tumor payudara (sesuai ERD laporan, Gambar 3.4).

Tiga entitas beserta atributnya (persis sesuai ERD):
1. data_pasien    : id_pasien (PK), fitur_mean (10), fitur_se (10), fitur_worst (10), waktu_input, diagnosis_asli.
2. model          : id_model (PK), nama_model, metode_interpretasi, accuracy, precision, recall, f1_score.
3. hasil_prediksi : id_prediksi (PK), label_prediksi, probabilitas_malignant, waktu_prediksi,
                    serta id_pasien (FK) dan id_model (FK) dari agregasi relasi "Diprediksi".

Relasi: data_pasien M : N model (relasi "Diprediksi"); setiap pasangan data_pasien-model
menghasilkan tepat satu hasil_prediksi (relasi "Menghasilkan", 1 : 1), sehingga pada tabel
hasil_prediksi pasangan (id_pasien, id_model) bersifat UNIQUE dan menjadi foreign key.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from preprocessing.data import FITUR

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "basis_data" / "riwayat_prediksi.db"

# Nama model baku di basis data (nama lain dari train.py / UI dipetakan ke sini)
NAMA_MODEL_BAKU = {
    "random forest": "Random Forest",
    "rf": "Random Forest",
    "support vector machine": "Support Vector Machine",
    "svm": "Support Vector Machine",
    "svm rbf": "Support Vector Machine",
}
METODE_INTERPRETASI = {"Random Forest": "Feature Importance", "Support Vector Machine": "SHAP KernelExplainer"}
LABEL_SAH = {"Benign", "Malignant"}


def nama_model_baku(nama: str) -> str:
    if not isinstance(nama, str) or not nama.strip():
        raise ValueError("Nama model tidak boleh kosong.")
    return NAMA_MODEL_BAKU.get(nama.strip().lower(), nama.strip())


def _sekarang() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _koneksi(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def inisialisasi_database(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Membuat tabel data_pasien, model, dan hasil_prediksi bila belum ada (dan memindahkan data versi lama)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kolom_fitur = ",\n                ".join(f"{f} REAL NOT NULL" for f in FITUR)
    with _koneksi(path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS data_pasien (
                id_pasien INTEGER PRIMARY KEY AUTOINCREMENT,
                {kolom_fitur},
                waktu_input TEXT NOT NULL,
                diagnosis_asli TEXT CHECK (diagnosis_asli IN ('Benign', 'Malignant'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model (
                id_model INTEGER PRIMARY KEY AUTOINCREMENT,
                nama_model TEXT NOT NULL UNIQUE,
                metode_interpretasi TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hasil_prediksi (
                id_prediksi INTEGER PRIMARY KEY AUTOINCREMENT,
                id_pasien INTEGER NOT NULL REFERENCES data_pasien(id_pasien) ON DELETE CASCADE,
                id_model INTEGER NOT NULL REFERENCES model(id_model),
                label_prediksi TEXT NOT NULL CHECK (label_prediksi IN ('Benign', 'Malignant')),
                probabilitas_malignant REAL NOT NULL CHECK (probabilitas_malignant BETWEEN 0 AND 1),
                waktu_prediksi TEXT NOT NULL,
                UNIQUE (id_pasien, id_model)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hasil_waktu ON hasil_prediksi(waktu_prediksi)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hasil_model ON hasil_prediksi(id_model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hasil_label ON hasil_prediksi(label_prediksi)")
        _hapus_kolom_di_luar_erd(conn)
        _pindahkan_data_lama(conn)
    return path


def _hapus_kolom_di_luar_erd(conn: sqlite3.Connection) -> None:
    """Basis data versi sebelumnya menyimpan kolom metode_input di data_pasien (tidak ada di ERD) -> dihapus."""
    kolom = {r[1] for r in conn.execute("PRAGMA table_info(data_pasien)")}
    if "metode_input" in kolom:
        conn.execute("ALTER TABLE data_pasien DROP COLUMN metode_input")


def _pindahkan_data_lama(conn: sqlite3.Connection) -> None:
    """Memindahkan isi tabel versi lama (prediction_history, model_metadata) ke struktur ERD, lalu menghapusnya."""
    tabel = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "model_metadata" in tabel:
        for nama, metode, acc, prec, rec, f1 in conn.execute(
                "SELECT model_name, interpretation_method, accuracy, precision, recall, f1_score FROM model_metadata").fetchall():
            _upsert_model(conn, nama, metode, {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        conn.execute("DROP TABLE model_metadata")
    if "prediction_history" in tabel:
        import json

        # Versi lama menyimpan satu baris per (pasien, model). Baris dengan fitur, waktu, dan label asli yang sama
        # adalah pasien yang sama yang diprediksi beberapa model -> disatukan menjadi satu data_pasien (relasi M : N).
        pasien = {}  # kunci -> [(id_pasien, {model yang sudah dipakai})]
        for waktu, model, label, p_m, asli, fitur_json in conn.execute(
                "SELECT created_at, model, classification, probability_malignant, actual_label, features_json "
                "FROM prediction_history ORDER BY id").fetchall():
            try:
                fitur = json.loads(fitur_json)
                nama = nama_model_baku(model)
                daftar = pasien.setdefault((fitur_json, waktu, asli), [])
                tujuan = next((d for d in daftar if nama not in d[1]), None)
                if tujuan is None:
                    tujuan = (_insert_pasien(conn, fitur, asli, waktu), set())
                    daftar.append(tujuan)
                _insert_hasil(conn, tujuan[0], nama, label, p_m, waktu)
                tujuan[1].add(nama)
            except Exception:
                continue  # baris lama yang rusak dilewati
        conn.execute("DROP TABLE prediction_history")


# ---------------------------------------------------------------------------
# Entitas model
# ---------------------------------------------------------------------------
def _angka(metrics: dict, nama: str):
    v = metrics.get(nama)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _upsert_model(conn, model_name, interpretation_method=None, metrics: dict | None = None) -> int:
    nama = nama_model_baku(model_name)
    metrics = metrics or {}
    metode = interpretation_method or METODE_INTERPRETASI.get(nama)
    if metrics:
        conn.execute(
            """
            INSERT INTO model (nama_model, metode_interpretasi, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(nama_model) DO UPDATE SET
                metode_interpretasi = excluded.metode_interpretasi, accuracy = excluded.accuracy,
                precision = excluded.precision, recall = excluded.recall, f1_score = excluded.f1_score
            """,
            (nama, metode, _angka(metrics, "accuracy"), _angka(metrics, "precision"),
             _angka(metrics, "recall"), _angka(metrics, "f1_score")),
        )
    else:
        ada = conn.execute("SELECT id_model FROM model WHERE nama_model = ?", (nama,)).fetchone()
        if ada:
            return int(ada[0])
        m = _metrik_dari_artefak(nama)
        conn.execute(
            "INSERT INTO model (nama_model, metode_interpretasi, accuracy, precision, recall, f1_score) VALUES (?, ?, ?, ?, ?, ?)",
            (nama, metode, m.get("accuracy"), m.get("precision"), m.get("recall"), m.get("f1_score")))
    return int(conn.execute("SELECT id_model FROM model WHERE nama_model = ?", (nama,)).fetchone()[0])


def _metrik_dari_artefak(nama: str) -> dict:
    """Rata-rata metrik Stratified 10-Fold x5 dari models/hasil_evaluasi.json (hasil train.py), jika tersedia."""
    import json

    path = ROOT / "models" / "hasil_evaluasi.json"
    try:
        for r in json.loads(path.read_text(encoding="utf-8")).get("kfold", []):
            if nama_model_baku(r["model"]) == nama:
                return {k: r.get(f"{k}_mean") for k in ("accuracy", "precision", "recall", "f1_score")}
    except Exception:
        pass
    return {}


def simpan_model(model_name: str, interpretation_method: str, metrics: dict,
                 db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Menyimpan atau memperbarui entitas model beserta nilai evaluasinya (dipanggil train.py)."""
    if not isinstance(metrics, dict):
        raise TypeError("metrics harus berupa dictionary metrik evaluasi.")
    path = inisialisasi_database(db_path)
    with _koneksi(path) as conn:
        return _upsert_model(conn, model_name, interpretation_method, metrics)


def baca_metadata_model(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    path = inisialisasi_database(db_path)
    with _koneksi(path) as conn:
        return pd.read_sql_query(
            "SELECT id_model, nama_model, metode_interpretasi, accuracy, precision, recall, f1_score "
            "FROM model ORDER BY id_model", conn)


# ---------------------------------------------------------------------------
# Entitas data_pasien dan hasil_prediksi
# ---------------------------------------------------------------------------
def _insert_pasien(conn, fitur: dict, diagnosis_asli: str | None, waktu: str | None = None) -> int:
    if not isinstance(fitur, dict) or any(f not in fitur for f in FITUR):
        raise ValueError("Sebanyak 30 fitur harus tersedia untuk disimpan.")
    if diagnosis_asli not in (None, "Benign", "Malignant"):
        diagnosis_asli = None
    kolom = ", ".join(FITUR)
    tanda = ", ".join("?" for _ in FITUR)
    cur = conn.execute(
        f"INSERT INTO data_pasien ({kolom}, waktu_input, diagnosis_asli) VALUES ({tanda}, ?, ?)",
        (*[float(fitur[f]) for f in FITUR], waktu or _sekarang(), diagnosis_asli),
    )
    return int(cur.lastrowid)


def _insert_hasil(conn, id_pasien: int, model: str, label: str, probabilitas_malignant: float, waktu: str | None = None) -> int:
    if label not in LABEL_SAH:
        raise ValueError("classification harus Benign atau Malignant.")
    id_model = _upsert_model(conn, model)
    cur = conn.execute(
        "INSERT INTO hasil_prediksi (id_pasien, id_model, label_prediksi, probabilitas_malignant, waktu_prediksi) "
        "VALUES (?, ?, ?, ?, ?)",
        (int(id_pasien), id_model, label, float(probabilitas_malignant), waktu or _sekarang()),
    )
    return int(cur.lastrowid)


def simpan_prediksi(
    model: str,
    classification: str,
    probability_benign: float | None,
    probability_malignant: float | None,
    input_method: str,
    fitur: dict,
    actual_label: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    verified: bool = False,
    id_pasien: int | None = None,
) -> int:
    """
    Menyimpan satu hasil prediksi dan mengembalikan id_prediksi.
    Jika id_pasien diberikan (pasien yang sama diprediksi model lain), data pasien tidak disimpan ulang.
    input_method tetap diterima agar pemanggil lama tidak rusak, tetapi tidak disimpan karena tidak ada di ERD.
    verified=False mencegah penyimpanan accidental/dummy.
    """
    if not verified:
        raise PermissionError("Record hanya boleh disimpan setelah prediksi backend terverifikasi.")
    if classification not in LABEL_SAH:
        raise ValueError("classification harus Benign atau Malignant.")
    if not isinstance(fitur, dict) or len(fitur) != 30:
        raise ValueError("Sebanyak 30 fitur harus disimpan.")
    if probability_malignant is None:
        if probability_benign is None:
            raise ValueError("Probabilitas prediksi wajib diisi.")
        probability_malignant = 1.0 - float(probability_benign)

    path = inisialisasi_database(db_path)
    with _koneksi(path) as conn:
        if id_pasien is None:
            id_pasien = _insert_pasien(conn, fitur, actual_label)
        return _insert_hasil(conn, id_pasien, model, classification, probability_malignant)


def id_pasien_dari_prediksi(id_prediksi: list[int] | int, db_path: str | Path = DEFAULT_DB_PATH):
    """Mengembalikan id_pasien untuk satu atau beberapa id_prediksi (urutan dipertahankan)."""
    tunggal = not isinstance(id_prediksi, (list, tuple))
    daftar = [int(id_prediksi)] if tunggal else [int(i) for i in id_prediksi]
    path = inisialisasi_database(db_path)
    with _koneksi(path) as conn:
        peta = {}
        for awal in range(0, len(daftar), 500):
            bagian = daftar[awal:awal + 500]
            q = f"SELECT id_prediksi, id_pasien FROM hasil_prediksi WHERE id_prediksi IN ({','.join('?' * len(bagian))})"
            peta.update(dict(conn.execute(q, bagian).fetchall()))
    hasil = [peta.get(i) for i in daftar]
    return hasil[0] if tunggal else hasil


# ---------------------------------------------------------------------------
# Pembacaan riwayat (gabungan tiga tabel)
# ---------------------------------------------------------------------------
_SELECT_RIWAYAT = """
    SELECT h.id_prediksi, h.waktu_prediksi, m.nama_model, h.label_prediksi,
           1.0 - h.probabilitas_malignant AS probabilitas_benign, h.probabilitas_malignant,
           p.waktu_input, p.diagnosis_asli, p.id_pasien
    FROM hasil_prediksi h
    JOIN data_pasien p ON p.id_pasien = h.id_pasien
    JOIN model m ON m.id_model = h.id_model
"""
_NAMA_KOLOM = {
    "id_prediksi": "ID", "waktu_prediksi": "Tanggal dan waktu", "nama_model": "Model",
    "label_prediksi": "Hasil klasifikasi", "probabilitas_benign": "Probabilitas Benign",
    "probabilitas_malignant": "Probabilitas Malignant", "waktu_input": "Waktu input",
    "diagnosis_asli": "Label aktual dataset", "id_pasien": "ID pasien",
}


def _where_filters(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    if filters.get("id_search"):
        clauses.append("CAST(h.id_prediksi AS TEXT) LIKE ?")
        params.append(f"%{filters['id_search']}%")
    if filters.get("model") and filters["model"] != "Semua":
        clauses.append("m.nama_model = ?")
        params.append(nama_model_baku(filters["model"]))
    if filters.get("classification") and filters["classification"] != "Semua":
        clauses.append("h.label_prediksi = ?")
        params.append(filters["classification"])
    if filters.get("tanggal_awal"):
        clauses.append("date(h.waktu_prediksi) >= date(?)")
        params.append(filters["tanggal_awal"])
    if filters.get("tanggal_akhir"):
        clauses.append("date(h.waktu_prediksi) <= date(?)")
        params.append(filters["tanggal_akhir"])
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


def baca_riwayat(filters: dict | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Membaca riwayat prediksi (gabungan hasil_prediksi, data_pasien, model), terbaru lebih dulu."""
    path = inisialisasi_database(db_path)
    where, params = _where_filters(filters or {})
    with _koneksi(path) as conn:
        data = pd.read_sql_query(_SELECT_RIWAYAT + where + " ORDER BY h.waktu_prediksi DESC, h.id_prediksi DESC",
                                 conn, params=params)
    return data.rename(columns=_NAMA_KOLOM) if not data.empty else data


def ambil_detail(id: int, db_path: str | Path = DEFAULT_DB_PATH) -> dict | None:
    """Mengambil satu hasil prediksi beserta 30 fitur data pasiennya berdasarkan id_prediksi."""
    path = inisialisasi_database(db_path)
    with _koneksi(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_SELECT_RIWAYAT + " WHERE h.id_prediksi = ?", (int(id),)).fetchone()
        if row is None:
            return None
        fitur_row = conn.execute(f"SELECT {', '.join(FITUR)} FROM data_pasien WHERE id_pasien = ?",
                                 (row["id_pasien"],)).fetchone()
    detail = {_NAMA_KOLOM[k]: row[k] for k in row.keys()}
    detail["Fitur"] = {f: float(fitur_row[f]) for f in FITUR}
    return detail
