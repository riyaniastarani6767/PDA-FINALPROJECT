from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "basis_data" / "riwayat_prediksi.db"


def inisialisasi_database(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Membuat database dan tabel riwayat serta metadata model bila belum ada."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                classification TEXT NOT NULL,
                probability_benign REAL,
                probability_malignant REAL,
                input_method TEXT NOT NULL,
                actual_label TEXT,
                features_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL UNIQUE,
                interpretation_method TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                tn INTEGER,
                fp INTEGER,
                fn INTEGER,
                tp INTEGER,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_created_at ON prediction_history(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_model ON prediction_history(model)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_classification ON prediction_history(classification)"
        )
    return path


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
) -> int:
    """Menyimpan satu hasil prediksi. verified=False mencegah penyimpanan accidental/dummy."""
    if not verified:
        raise PermissionError("Record hanya boleh disimpan setelah prediksi backend terverifikasi.")
    if classification not in {"Benign", "Malignant"}:
        raise ValueError("classification harus Benign atau Malignant.")
    if not isinstance(fitur, dict) or len(fitur) != 30:
        raise ValueError("Sebanyak 30 fitur harus disimpan sebagai JSON.")

    path = inisialisasi_database(db_path)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with sqlite3.connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history
                (created_at, model, classification, probability_benign,
                 probability_malignant, input_method, actual_label, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                model,
                classification,
                probability_benign,
                probability_malignant,
                input_method,
                actual_label,
                json.dumps(fitur, ensure_ascii=False),
            ),
        )
    return int(cursor.lastrowid)


def _where_filters(filters: dict) -> tuple[str, list]:
    clauses = []
    params = []

    if filters.get("id_search"):
        clauses.append("CAST(id AS TEXT) LIKE ?")
        params.append(f"%{filters['id_search']}%")
    if filters.get("model") and filters["model"] != "Semua":
        clauses.append("model = ?")
        params.append(filters["model"])
    if filters.get("classification") and filters["classification"] != "Semua":
        clauses.append("classification = ?")
        params.append(filters["classification"])
    if filters.get("tanggal_awal"):
        clauses.append("date(created_at) >= date(?)")
        params.append(filters["tanggal_awal"])
    if filters.get("tanggal_akhir"):
        clauses.append("date(created_at) <= date(?)")
        params.append(filters["tanggal_akhir"])

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def baca_riwayat(filters: dict | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Membaca riwayat dengan filter, selalu diurutkan waktu terbaru."""
    path = inisialisasi_database(db_path)
    filters = filters or {}
    where, params = _where_filters(filters)

    with sqlite3.connect(path) as conn:
        data = pd.read_sql_query(
            """
            SELECT id, created_at, model, classification,
                   probability_benign, probability_malignant, input_method, actual_label
            FROM prediction_history
            """ + where + " ORDER BY created_at DESC, id DESC",
            conn,
            params=params,
        )

    if not data.empty:
        data = data.rename(columns={
            "id": "ID",
            "created_at": "Tanggal dan waktu",
            "model": "Model",
            "classification": "Hasil klasifikasi",
            "probability_benign": "Probabilitas Benign",
            "probability_malignant": "Probabilitas Malignant",
            "input_method": "Metode input",
            "actual_label": "Label aktual dataset",
        })
    return data


def ambil_detail(id: int, db_path: str | Path = DEFAULT_DB_PATH) -> dict | None:
    """Mengambil satu record beserta seluruh fitur JSON berdasarkan ID."""
    path = inisialisasi_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT id, created_at, model, classification,
                   probability_benign, probability_malignant,
                   input_method, actual_label, features_json
            FROM prediction_history
            WHERE id = ?
            """,
            (int(id),),
        ).fetchone()

    if row is None:
        return None

    try:
        fitur = json.loads(row[8])
    except json.JSONDecodeError:
        fitur = {}

    return {
        "ID": row[0],
        "Tanggal dan waktu": row[1],
        "Model": row[2],
        "Hasil klasifikasi": row[3],
        "Probabilitas Benign": row[4],
        "Probabilitas Malignant": row[5],
        "Metode input": row[6],
        "Label aktual dataset": row[7],
        "Fitur": fitur,
    }


def simpan_model(
    model_name: str,
    interpretation_method: str,
    metrics: dict,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """
    Menyimpan atau memperbarui metadata model berdasarkan identitas model.

    Fungsi ini tidak memuat, menulis, atau menimpa file model .pkl.
    Interface sengaja kompatibel dengan pemanggilan train.py:
    simpan_model(model_name, interpretation_method, metrics).
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name harus berupa nama model yang tidak kosong.")
    if not isinstance(metrics, dict):
        raise TypeError("metrics harus berupa dictionary metrik evaluasi.")

    def angka(nama: str):
        value = metrics.get(nama)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def integer(nama: str):
        value = metrics.get(nama)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    path = inisialisasi_database(db_path)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with sqlite3.connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_metadata
                (model_name, interpretation_method, accuracy, precision, recall,
                 f1_score, tn, fp, fn, tp, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                interpretation_method = excluded.interpretation_method,
                accuracy = excluded.accuracy,
                precision = excluded.precision,
                recall = excluded.recall,
                f1_score = excluded.f1_score,
                tn = excluded.tn,
                fp = excluded.fp,
                fn = excluded.fn,
                tp = excluded.tp,
                updated_at = excluded.updated_at
            """,
            (
                model_name.strip(),
                interpretation_method,
                angka("accuracy"),
                angka("precision"),
                angka("recall"),
                angka("f1_score"),
                integer("TN") if integer("TN") is not None else integer("tn"),
                integer("FP") if integer("FP") is not None else integer("fp"),
                integer("FN") if integer("FN") is not None else integer("fn"),
                integer("TP") if integer("TP") is not None else integer("tp"),
                updated_at,
            ),
        )

        row = conn.execute(
            "SELECT id FROM model_metadata WHERE model_name = ?",
            (model_name.strip(),),
        ).fetchone()
    return int(row[0])


def baca_metadata_model(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Membaca semua metadata model, diurutkan berdasarkan nama model."""
    path = inisialisasi_database(db_path)
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(
            """
            SELECT model_name, interpretation_method, accuracy, precision,
                   recall, f1_score, tn, fp, fn, tp, updated_at
            FROM model_metadata
            ORDER BY model_name ASC
            """,
            conn,
        )
