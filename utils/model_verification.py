from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing.data import FITUR


VERIFICATION_PROCESS_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
VERIFICATION_PATH = MODELS_DIR / "verified_models.json"
MANIFEST_PATH = MODELS_DIR / "preprocessing_manifest.json"
DATA_PATH = ROOT / "data" / "breast_cancer.csv"

MODEL_PATHS = {
    "Random Forest": MODELS_DIR / "model_rf.pkl",
    "Support Vector Machine": MODELS_DIR / "model_svm.pkl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_contract_fingerprint() -> str:
    payload = json.dumps(
        {"feature_count": len(FITUR), "feature_order": list(FITUR),
         "positive_class": "Malignant", "positive_class_value": 1,
         "missing_value_policy": "reject"},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def invalidate_marker(path: Path = VERIFICATION_PATH) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def atomic_write_marker(marker: dict, path: Path = VERIFICATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".verified_models.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def load_and_validate_marker(path: Path = VERIFICATION_PATH,
                             root: Path = ROOT,
                             model_paths: dict[str, Path] | None = None,
                             manifest_path: Path | None = None) -> tuple[bool, str]:
    model_paths = model_paths or {
        "Random Forest": root / "models" / "model_rf.pkl",
        "Support Vector Machine": root / "models" / "model_svm.pkl",
    }
    manifest_path = manifest_path or root / "models" / "preprocessing_manifest.json"

    if not path.exists():
        return False, "Verification marker tidak tersedia."
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Verification marker tidak valid JSON: {exc}"

    required = {
        "verification_status", "verification_timestamp", "python_version",
        "verification_process_version", "model_artifacts", "preprocessing_manifest",
        "canonical_feature_count", "canonical_feature_contract_fingerprint",
        "positive_class_mapping", "missing_value_policy", "models",
        "test_summary",
    }
    if not required.issubset(marker):
        return False, "Struktur verification marker tidak lengkap."
    if marker["verification_status"] != "verified":
        return False, "Verification marker tidak berstatus verified."
    if marker["verification_process_version"] != VERIFICATION_PROCESS_VERSION:
        return False, "Versi verification process tidak cocok."
    if marker["canonical_feature_count"] != 30:
        return False, "Jumlah canonical feature tidak cocok."
    if marker["canonical_feature_contract_fingerprint"] != canonical_contract_fingerprint():
        return False, "Canonical feature contract fingerprint berubah."
    if marker["missing_value_policy"] != "REJECT":
        return False, "Missing-value policy tidak cocok."
    if marker["positive_class_mapping"] != {"Benign": 0, "Malignant": 1}:
        return False, "Positive-class mapping tidak cocok."

    if not manifest_path.exists():
        return False, "Preprocessing manifest tidak tersedia."
    manifest_entry = marker["preprocessing_manifest"]
    if manifest_entry.get("sha256") != sha256_file(manifest_path):
        return False, "Fingerprint preprocessing manifest berubah."

    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            return False, f"Artefak model {model_name} tidak tersedia."
        entry = marker["models"].get(model_name)
        artifact_entry = marker["model_artifacts"].get(model_name)
        if not entry or not entry.get("status") == "verified":
            return False, f"Status verifikasi {model_name} tidak valid."
        if not artifact_entry or artifact_entry.get("sha256") != sha256_file(model_path):
            return False, f"Fingerprint artefak {model_name} berubah."

    return True, "Verification marker valid."


def _load_payload(path: Path, model_name: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"Artefak model tidak ditemukan: {path}")
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Artefak model rusak atau tidak dapat dimuat: {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Payload artifact {model_name} bukan dictionary.")
    if payload.get("fitur") != list(FITUR):
        raise RuntimeError(f"Urutan fitur payload {model_name} tidak cocok dengan FITUR canonical.")
    model = payload.get("model")
    scaler = payload.get("scaler")
    if model is None or scaler is None:
        raise RuntimeError(f"Payload {model_name} tidak memuat model atau scaler.")
    if not callable(getattr(model, "predict", None)) or not callable(getattr(model, "predict_proba", None)):
        raise RuntimeError(f"Model {model_name} tidak menyediakan predict() dan predict_proba().")
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        raise RuntimeError(f"Scaler {model_name} belum fitted.")
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if mean.shape != (30,) or scale.shape != (30,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise RuntimeError(f"Scaler {model_name} tidak valid untuk 30 fitur.")
    return payload


def run_verification(root: Path = ROOT,
                     write_marker: bool = True,
                     model_paths: dict[str, Path] | None = None,
                     manifest_path: Path | None = None,
                     data_path: Path | None = None) -> dict:
    root = root.resolve()
    model_paths = model_paths or {
        "Random Forest": root / "models" / "model_rf.pkl",
        "Support Vector Machine": root / "models" / "model_svm.pkl",
    }
    manifest_path = manifest_path or root / "models" / "preprocessing_manifest.json"
    marker_path = root / "models" / "verified_models.json"
    data_path = data_path or root / "data" / "breast_cancer.csv"

    # A write-verification invalidates stale trust before checks; dry runs leave existing markers intact.
    if write_marker:
        invalidate_marker(marker_path)

    checks: dict[str, dict] = {
        "artifact_integrity": {"status": "pending"},
        "model_identity": {"status": "pending"},
        "single_inference": {"status": "pending"},
        "batch_inference": {"status": "pending"},
        "invalid_input": {"status": "pending"},
        "database_integration": {"status": "pending"},
    }

    try:
        if not manifest_path.exists():
            raise RuntimeError("Preprocessing manifest tidak tersedia.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_manifest = {
            "manifest_version", "feature_names", "feature_count", "target_mapping",
            "positive_class", "missing_value_policy", "preprocessing_order",
        }
        if not required_manifest.issubset(manifest):
            raise RuntimeError("Preprocessing manifest tidak memiliki field wajib.")
        if manifest["feature_names"] != list(FITUR) or manifest["feature_count"] != 30:
            raise RuntimeError("Preprocessing manifest tidak cocok dengan FITUR canonical.")
        if manifest["target_mapping"] != {"B": 0, "M": 1}:
            raise RuntimeError("Target mapping pada manifest tidak cocok.")
        if manifest["positive_class"] != "Malignant":
            raise RuntimeError("Positive class pada manifest bukan Malignant.")
        if manifest["missing_value_policy"] != "reject":
            raise RuntimeError("Missing-value policy pada manifest tidak REJECT.")

        payloads = {}
        model_entries = {}
        artifact_entries = {}
        for model_name, path in model_paths.items():
            payload = _load_payload(path, model_name)
            payloads[model_name] = payload
            artifact_entries[model_name] = {"sha256": sha256_file(path), "path": str(path)}

        rf_payload = payloads["Random Forest"]
        svm_payload = payloads["Support Vector Machine"]
        rf = rf_payload["model"]
        svm = svm_payload["model"]

        rf_type = f"{type(rf).__module__}.{type(rf).__name__}"
        svm_type = f"{type(svm).__module__}.{type(svm).__name__}"
        if type(rf).__name__ != "RandomForestManual":
            raise RuntimeError("Tipe model Random Forest tidak sesuai.")
        if type(svm).__name__ != "SVMManual":
            raise RuntimeError("Tipe model SVM tidak sesuai.")
        if rf.n_estimators != 100 or rf.max_depth is not None or rf.min_samples_split != 2 or rf.max_features != "sqrt" or rf.random_state != 42:
            raise RuntimeError("Konfigurasi Random Forest tidak sesuai konfigurasi terlatih.")
        if svm.C != 1.0 or svm.gamma != "scale" or not getattr(svm, "probability", False):
            raise RuntimeError("Konfigurasi SVM RBF tidak sesuai konfigurasi terlatih.")
        if not callable(getattr(svm, "decision_function", None)):
            raise RuntimeError("SVM tidak menyediakan decision_function().")
        if not hasattr(svm, "prob_a_") or not hasattr(svm, "prob_b_"):
            raise RuntimeError("SVM tidak memuat parameter Platt Scaling terlatih.")

        checks["artifact_integrity"] = {
            "status": "passed", "payloads_loaded": True,
            "manifest_valid": True, "scalers_fitted": True,
        }

        model_entries["Random Forest"] = {
            "status": "verified", "model_type": rf_type,
            "n_features": int(getattr(rf, "n_features_", 30)),
            "configuration": {"n_estimators": rf.n_estimators, "max_depth": rf.max_depth,
                              "min_samples_split": rf.min_samples_split,
                              "max_features": rf.max_features, "random_state": rf.random_state},
        }
        model_entries["Support Vector Machine"] = {
            "status": "verified", "model_type": svm_type,
            "C": float(svm.C), "gamma": svm.gamma,
            "gamma_value": float(svm.gamma_), "probability": True,
            "n_features": int(svm.n_features_), "decision_function": True,
        }
        checks["model_identity"] = {"status": "passed", "positive_class": "Malignant=1"}

        # Import lazily to avoid circular import during backend initialization.
        from utils import backend_adapter

        if not data_path.exists():
            raise RuntimeError("Dataset verification tidak ditemukan.")
        dataset = pd.read_csv(data_path)
        representative_indices = [0, 99, 568]
        representative = dataset.iloc[representative_indices]

        single_results = {}
        batch_results = {}
        for model_name in ("Random Forest", "Support Vector Machine"):
            single_results[model_name] = []
            for index in representative_indices:
                features = {name: float(dataset.iloc[index][name]) for name in FITUR}
                response = backend_adapter.predict_single(features, model_name, "csv")
                if response["status"] != "success":
                    raise RuntimeError(f"Single inference {model_name} gagal: {response.get('error')}")
                benign = float(response["probability_benign"])
                malignant = float(response["probability_malignant"])
                if response["classification"] not in (0, 1):
                    raise RuntimeError(f"Classification {model_name} tidak valid.")
                if not np.isfinite([benign, malignant]).all() or not 0 <= benign <= 1 or not 0 <= malignant <= 1:
                    raise RuntimeError(f"Probability {model_name} tidak valid.")
                if not np.isclose(benign + malignant, 1.0, rtol=1e-7, atol=1e-8):
                    raise RuntimeError(f"Jumlah probability {model_name} tidak sama dengan 1.")
                if response["positive_class"] != "Malignant":
                    raise RuntimeError(f"Positive class {model_name} tidak cocok.")
                single_results[model_name].append(response)

            batch = backend_adapter.predict_batch(dataset[FITUR].iloc[representative_indices].copy(), model_name, "csv")
            if batch["status"] != "success" or batch["jumlah_berhasil"] != 3:
                raise RuntimeError(f"Batch inference {model_name} gagal.")
            for offset, detail in enumerate(batch["detail"]):
                single = single_results[model_name][offset]
                if detail["classification"] != single["classification"]:
                    raise RuntimeError(f"Classification batch/single {model_name} tidak konsisten.")
                if not np.isclose(detail["probability_benign"], single["probability_benign"]):
                    raise RuntimeError(f"Probability Benign batch/single {model_name} tidak konsisten.")
                if not np.isclose(detail["probability_malignant"], single["probability_malignant"]):
                    raise RuntimeError(f"Probability Malignant batch/single {model_name} tidak konsisten.")
            batch_results[model_name] = {"rows": 3, "all_valid": True,
                                         "source_indices": [0, 99, 568]}

        checks["single_inference"] = {"status": "passed", "samples_per_model": 3,
                                      "probability_contract": "passed"}
        checks["batch_inference"] = {"status": "passed", "samples_per_model": 3,
                                     "single_vs_batch_consistent": True}

        svm_sample = single_results["Support Vector Machine"][1]
        if "decision_score" not in svm_sample:
            raise RuntimeError("SVM decision score tidak tersedia.")
        model_entries["Support Vector Machine"]["platt_scaling"] = "calibrated_predict_proba"
        model_entries["Support Vector Machine"]["hard_prediction_source"] = "predict()"
        model_entries["Support Vector Machine"]["probability_source"] = "predict_proba()"

        # Invalid input contract.
        sample_features = {name: float(dataset.iloc[0][name]) for name in FITUR}
        invalid_cases = []
        missing_value = dict(sample_features)
        missing_value[FITUR[0]] = np.nan
        invalid_cases.append(("missing_value", backend_adapter.predict_single(missing_value, "Random Forest", "manual")))
        missing_feature = dict(sample_features)
        missing_feature.pop(FITUR[0])
        invalid_cases.append(("missing_feature", backend_adapter.predict_single(missing_feature, "Random Forest", "manual")))
        extra_feature = dict(sample_features)
        extra_feature["extra_feature"] = 1.0
        invalid_cases.append(("extra_feature", backend_adapter.predict_single(extra_feature, "Random Forest", "manual")))
        invalid_cases.append(("invalid_model", backend_adapter.predict_single(sample_features, "Unknown Model", "manual")))
        if any(response["status"] != "error" for _, response in invalid_cases):
            raise RuntimeError("Kontrak input invalid tidak ditolak sepenuhnya.")
        checks["invalid_input"] = {"status": "passed", "cases": [name for name, _ in invalid_cases]}

        # Database integration is tested only against a temporary SQLite database.
        from basis_data.database import ambil_detail, baca_riwayat, inisialisasi_database, simpan_prediksi
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            temp_db = Path(temp_dir) / "verification.db"
            db = inisialisasi_database(temp_db)
            success_response = single_results["Random Forest"][0]
            try:
                simpan_prediksi(
                    model="Random Forest", classification=success_response["classification_label"],
                    probability_benign=success_response["probability_benign"],
                    probability_malignant=success_response["probability_malignant"],
                    input_method="manual", fitur=sample_features,
                    db_path=db, verified=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Simpan prediksi sukses gagal diuji: {exc}") from exc
            if baca_riwayat(db_path=db).empty:
                raise RuntimeError("Prediksi sukses tidak dapat dibaca dari database sementara.")
            try:
                simpan_prediksi(
                    model="Random Forest", classification="Benign",
                    probability_benign=None, probability_malignant=None,
                    input_method="manual", fitur=sample_features,
                    db_path=db, verified=False,
                )
                raise RuntimeError("Prediksi unverified berhasil disimpan.")
            except PermissionError:
                pass
        checks["database_integration"] = {"status": "passed", "production_db_used": False,
                                           "temp_db_success": True, "unverified_save_blocked": True}

        marker = {
            "verification_status": "verified",
            "verification_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "python_version": sys.version.split()[0],
            "verification_process_version": VERIFICATION_PROCESS_VERSION,
            "model_artifacts": artifact_entries,
            "preprocessing_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "canonical_feature_count": 30,
            "canonical_feature_order": list(FITUR),
            "canonical_feature_contract_fingerprint": canonical_contract_fingerprint(),
            "positive_class_mapping": {"Benign": 0, "Malignant": 1},
            "missing_value_policy": "REJECT",
            "models": model_entries,
            "test_summary": {
                "artifact_integrity": "passed",
                "single_inference": "passed",
                "batch_inference": "passed",
                "invalid_input": "passed",
                "database_integration": "passed",
                "complete_pytest": "pending_after_verification",
            },
            "checks": checks,
        }

        if write_marker:
            atomic_write_marker(marker, marker_path)
        return marker

    except Exception as exc:
        if write_marker:
            invalidate_marker(marker_path)
        return {
            "verification_status": "failed",
            "verification_error": str(exc),
            "checks": checks,
        }
