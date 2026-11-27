import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
GAMBAR = ROOT / "gambar"
METRICS = ["accuracy", "precision", "recall", "f1_score"]


def muat_hasil_evaluasi() -> dict:
    path = MODELS / "hasil_evaluasi.json"
    if not path.exists():
        return {"available": False, "reason": "models/hasil_evaluasi.json belum tersedia."}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"hasil_evaluasi.json tidak dapat dibaca: {exc}"}

    holdout = data.get("holdout")
    kfold = data.get("kfold")
    if not isinstance(holdout, dict) or not isinstance(kfold, (list, type(None))):
        return {"available": False, "reason": "Format hasil_evaluasi.json tidak sesuai artefak training."}
    return {"available": True, "holdout": holdout, "kfold": kfold}


def muat_kfold_raw() -> tuple[pd.DataFrame | None, str | None]:
    path = MODELS / "hasil_kfold.csv"
    if not path.exists():
        return None, "models/hasil_kfold.csv belum tersedia."
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return None, f"hasil_kfold.csv tidak dapat dibaca: {exc}"

    required = {"model", "ulangan", "fold", *METRICS}
    if not required.issubset(data.columns):
        return None, "Kolom hasil_kfold.csv tidak sesuai format ev.validasi_kfold()."
    return data, None


def muat_feature_importance() -> tuple[pd.DataFrame | None, str | None]:
    path = MODELS / "feature_importance_rf.csv"
    if not path.exists():
        return None, "models/feature_importance_rf.csv belum tersedia."
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return None, f"feature_importance_rf.csv tidak dapat dibaca: {exc}"

    required = {"peringkat", "fitur", "nama", "importance"}
    if not required.issubset(data.columns):
        return None, "Format feature_importance_rf.csv tidak sesuai tabel_feature_importance()."
    return data, None


def muat_ranking_shap() -> tuple[pd.DataFrame | None, str | None]:
    path = MODELS / "ranking_shap_svm.csv"
    if not path.exists():
        return None, "models/ranking_shap_svm.csv belum tersedia."
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return None, f"ranking_shap_svm.csv tidak dapat dibaca: {exc}"

    required = {"peringkat", "fitur", "nama", "mean_abs_shap"}
    if not required.issubset(data.columns):
        return None, "Format ranking_shap_svm.csv tidak sesuai ranking_global()."
    return data, None


def muat_shap_global() -> tuple[dict | None, str | None]:
    path = MODELS / "shap_global.pkl"
    if not path.exists():
        return None, "models/shap_global.pkl belum tersedia."
    try:
        data = joblib.load(path)
    except Exception as exc:
        return None, f"shap_global.pkl tidak dapat dibaca: {exc}"

    if not isinstance(data, dict):
        return None, "Format shap_global.pkl bukan dict artefak train.py."
    required = {"shap", "X_asli", "ranking", "base_value"}
    if not required.issubset(data):
        return None, "Isi shap_global.pkl tidak sesuai artefak train.py."
    return data, None


def path_gambar(nama: str) -> Path | None:
    path = GAMBAR / nama
    return path if path.exists() else None
