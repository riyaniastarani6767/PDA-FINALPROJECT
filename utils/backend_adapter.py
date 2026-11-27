from enum import Enum
from pathlib import Path

import hashlib
import json
import uuid

import joblib
import numpy as np
import pandas as pd

from preprocessing.data import FITUR
from utils.model_verification import load_and_validate_marker


class BackendStatus(str, Enum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_VERIFIED = "MODEL_NOT_VERIFIED"
    READY = "READY"
    PREDICTION_ERROR = "PREDICTION_ERROR"


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "Random Forest": ROOT / "models" / "model_rf.pkl",
    "Support Vector Machine": ROOT / "models" / "model_svm.pkl",
}
VERIFICATION_PATH = ROOT / "models" / "verified_models.json"
PREPROCESSING_MANIFEST_PATH = ROOT / "models" / "preprocessing_manifest.json"


class PreprocessingContractError(ValueError):
    """Kesalahan validasi atau konfigurasi kontrak preprocessing."""


class ModelLoadError(RuntimeError):
    """Kesalahan pemuatan atau validasi artifact model."""


class InferenceError(RuntimeError):
    """Kesalahan saat menjalankan inferensi pada model aktual."""


def muat_preprocessing_manifest() -> dict:
    """Memuat manifest preprocessing dari artefak JSON yang tersimpan."""
    if not PREPROCESSING_MANIFEST_PATH.exists():
        raise PreprocessingContractError(
            "Manifest preprocessing belum tersedia: models/preprocessing_manifest.json"
        )

    try:
        manifest = json.loads(PREPROCESSING_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreprocessingContractError("Manifest preprocessing bukan JSON valid.") from exc

    required = {
        "manifest_version",
        "training_artifact_version",
        "feature_names",
        "feature_count",
        "target_mapping",
        "positive_class",
        "missing_value_policy",
        "negative_value_policy",
        "preprocessing_order",
        "scaler_type",
        "scaler_artifact_reference",
        "model_artifacts",
        "training_preprocessing_reference",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise PreprocessingContractError(
            "Manifest preprocessing tidak lengkap: " + ", ".join(missing)
        )

    return manifest


def muat_scaler_terlatih(nama_model: str):
    """Memuat scaler yang sudah tersimpan pada payload model; tidak melakukan fit ulang."""
    path = MODEL_PATHS.get(nama_model)
    if path is None:
        raise PreprocessingContractError("Nama model tidak dikenali untuk preprocessing.")
    if not path.exists():
        raise PreprocessingContractError(f"Artefak model belum tersedia: {path}")

    payload = joblib.load(path)
    scaler = payload.get("scaler")
    fitur_artifak = payload.get("fitur")
    if scaler is None:
        raise PreprocessingContractError("Payload model tidak memuat scaler training.")
    if fitur_artifak != list(FITUR):
        raise PreprocessingContractError("Urutan fitur pada payload model tidak sama dengan FITUR canonical.")

    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        raise PreprocessingContractError("Scaler artifact belum berisi mean_ dan scale_.")
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if mean.shape != (len(FITUR),) or scale.shape != (len(FITUR),):
        raise PreprocessingContractError("Mean atau scale scaler tidak berukuran 30 fitur.")

    return scaler


def _normalisasi_nama_kolom(nama: object) -> str:
    return str(nama).strip().lower().replace(" ", "_")


def _urutkan_fitur_dict(fitur: dict) -> np.ndarray:
    if not isinstance(fitur, dict):
        raise PreprocessingContractError("Input satu pasien harus berupa mapping nama-fitur.")

    normalized = {}
    for nama, nilai in fitur.items():
        key = _normalisasi_nama_kolom(nama)
        if key in normalized:
            raise PreprocessingContractError(
                f"Nama kolom duplikat setelah normalisasi: {key}"
            )
        normalized[key] = nilai

    missing = [nama for nama in FITUR if nama not in normalized]
    if missing:
        raise PreprocessingContractError(
            f"Fitur wajib hilang: {len(missing)} fitur ({', '.join(missing[:5])})."
        )

    extra = [nama for nama in normalized if nama not in FITUR]
    if extra:
        raise PreprocessingContractError(
            f"Fitur tambahan tidak diperbolehkan: {len(extra)} fitur ({', '.join(extra[:5])})."
        )

    ordered = [normalized[nama] for nama in FITUR]
    try:
        return np.asarray(ordered, dtype=float).reshape(1, -1)
    except (TypeError, ValueError) as exc:
        raise PreprocessingContractError(
            "Semua nilai fitur harus berupa angka numerik."
        ) from exc


def _validasi_array_numerik(array: np.ndarray) -> None:
    if np.isnan(array).any():
        raise PreprocessingContractError("Missing value ditolak oleh kebijakan preprocessing.")
    if np.isinf(array).any():
        raise PreprocessingContractError("Nilai tak-hingga tidak diperbolehkan.")
    if (array < 0).any():
        raise PreprocessingContractError("Nilai negatif tidak diperbolehkan.")


def preprocess_single(nama_model: str, fitur_pasien: dict) -> np.ndarray:
    muat_preprocessing_manifest()
    raw = _urutkan_fitur_dict(fitur_pasien)

    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise PreprocessingContractError("Semua nilai fitur harus berupa angka numerik.") from exc

    _validasi_array_numerik(numeric)
    scaler = muat_scaler_terlatih(nama_model)
    return scaler.transform(numeric)


def preprocess_batch(nama_model: str, data: pd.DataFrame) -> np.ndarray:
    muat_preprocessing_manifest()
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise PreprocessingContractError("Data batch harus berupa DataFrame yang tidak kosong.")

    df = data.copy()
    normalized_names = [_normalisasi_nama_kolom(nama) for nama in df.columns]
    if len(normalized_names) != len(set(normalized_names)):
        duplicate_names = sorted(
            {nama for nama in normalized_names if normalized_names.count(nama) > 1}
        )
        raise PreprocessingContractError(
            "Nama kolom duplikat setelah normalisasi: " + ", ".join(duplicate_names)
        )
    df.columns = normalized_names

    missing = [nama for nama in FITUR if nama not in df.columns]
    if missing:
        raise PreprocessingContractError(
            f"Fitur wajib hilang: {len(missing)} fitur ({', '.join(missing[:5])})."
        )

    extra = [nama for nama in df.columns if nama not in FITUR]
    if extra:
        raise PreprocessingContractError(
            f"Fitur tambahan tidak diperbolehkan: {len(extra)} fitur ({', '.join(extra[:5])})."
        )

    raw = df[FITUR].copy()
    try:
        numeric = raw.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise PreprocessingContractError("Semua nilai fitur batch harus berupa angka numerik.") from exc

    numeric = np.asarray(numeric, dtype=float)
    _validasi_array_numerik(numeric)
    scaler = muat_scaler_terlatih(nama_model)
    return scaler.transform(numeric)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _serialize_source_index(value) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_model(model_name: str) -> dict:
    """Memuat dan memvalidasi payload model aktual beserta scaler tanpa training atau fit ulang."""
    if model_name not in MODEL_PATHS:
        raise ModelLoadError("Nama model tidak dikenali. Gunakan 'Random Forest' atau 'Support Vector Machine'.")

    path = MODEL_PATHS[model_name]
    if not path.exists():
        raise ModelLoadError(f"Artefak model tidak ditemukan: {path}")

    try:
        manifest = muat_preprocessing_manifest()
    except PreprocessingContractError as exc:
        raise ModelLoadError(str(exc)) from exc

    if manifest.get("feature_names") != list(FITUR):
        raise ModelLoadError("Manifest feature_names tidak sama dengan urutan canonical FITUR.")
    if manifest.get("feature_count") != len(FITUR):
        raise ModelLoadError("Manifest feature_count tidak sesuai jumlah canonical 30 fitur.")

    filename = path.name
    checksum_manifest = manifest.get("artifact_checksums", {}).get(filename)
    if checksum_manifest:
        try:
            actual_checksum = _sha256_file(path)
        except OSError as exc:
            raise ModelLoadError(f"Artifact checksum tidak dapat dibaca: {filename}") from exc
        if actual_checksum != checksum_manifest:
            raise ModelLoadError(f"Checksum artifact tidak sama dengan manifest: {filename}")

    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Artifact model rusak atau tidak dapat dimuat: {filename}") from exc

    if not isinstance(payload, dict):
        raise ModelLoadError("Payload artifact harus berupa dictionary.")

    model = payload.get("model")
    scaler = payload.get("scaler")
    artifact_features = payload.get("fitur")

    if model is None:
        raise ModelLoadError("Payload model tidak memuat objek 'model'.")
    if scaler is None:
        raise ModelLoadError("Payload model tidak memuat scaler training.")
    if artifact_features != list(FITUR):
        raise ModelLoadError("Urutan fitur payload model tidak sama dengan canonical FITUR.")

    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        raise ModelLoadError("Scaler payload tidak berisi mean_ dan scale_.")

    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if mean.shape != (len(FITUR),) or scale.shape != (len(FITUR),):
        raise ModelLoadError("Scaler payload tidak berukuran 30 fitur.")

    if not callable(getattr(model, "predict", None)):
        raise ModelLoadError("Objek model tidak menyediakan predict().")
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelLoadError("Objek model tidak menyediakan predict_proba().")

    if model_name == "Support Vector Machine":
        if not callable(getattr(model, "decision_function", None)):
            raise ModelLoadError("Model SVM tidak menyediakan decision_function().")

    return {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "feature_names": list(FITUR),
        "payload": payload,
        "artifact_path": str(path),
        "status": "loaded_not_verified",
    }


def _normalize_input_method(input_method: str) -> str:
    normalized = str(input_method).strip().lower()
    aliases = {
        "manual": "manual",
        "input_manual": "manual",
        "input manual": "manual",
        "sample": "sample",
        "dataset": "sample",
        "pilih_sampel_dataset": "sample",
        "csv": "csv",
        "upload_csv": "csv",
    }
    if normalized not in aliases:
        raise InferenceError("input_method harus 'manual', 'sample', atau 'csv'.")
    return aliases[normalized]


def _resolve_probability_mapping(model) -> dict:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return {0: 0, 1: 1}

    classes_array = np.asarray(classes).reshape(-1)
    if classes_array.size != 2:
        raise InferenceError("Predict_proba tidak mengembalikan tepat dua kelas.")

    column_map = {}
    for column, class_value in enumerate(classes_array.tolist()):
        try:
            numeric_class = int(class_value)
        except (TypeError, ValueError) as exc:
            raise InferenceError("Label kelas pada model bukan 0 atau 1.") from exc
        if numeric_class not in (0, 1):
            raise InferenceError("Model mengandung label kelas selain 0 dan 1.")
        if numeric_class in column_map:
            raise InferenceError("Model memiliki label kelas duplikat.")
        column_map[numeric_class] = column

    if set(column_map) != {0, 1}:
        raise InferenceError("Mapping predict_proba tidak memuat Benign=0 dan Malignant=1.")
    return column_map


def _probability_result(model, transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        probabilities = np.asarray(model.predict_proba(transformed), dtype=float)
    except Exception as exc:
        raise InferenceError("predict_proba() gagal dijalankan pada artifact aktual.") from exc

    if probabilities.ndim != 2 or probabilities.shape[0] != transformed.shape[0] or probabilities.shape[1] != 2:
        raise InferenceError("predict_proba() harus menghasilkan matriks (jumlah_baris, 2).")
    if not np.isfinite(probabilities).all():
        raise InferenceError("predict_proba() mengembalikan nilai tidak terbatas atau NaN.")
    if (probabilities < -1e-12).any() or (probabilities > 1.0 + 1e-12).any():
        raise InferenceError("predict_proba() mengembalikan nilai di luar rentang [0, 1].")

    probability_sum = probabilities.sum(axis=1)
    if not np.allclose(probability_sum, 1.0, rtol=1e-7, atol=1e-8):
        raise InferenceError("Jumlah probability per baris tidak sama dengan 1.")

    column_map = _resolve_probability_mapping(model)
    decision = None
    if callable(getattr(model, "decision_function", None)):
        try:
            decision = np.asarray(model.decision_function(transformed), dtype=float).reshape(-1)
            if decision.shape[0] != transformed.shape[0]:
                raise InferenceError("decision_function() menghasilkan jumlah baris yang tidak sesuai.")
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("decision_function() gagal dijalankan pada artifact aktual.") from exc

    mapped = np.column_stack([
        probabilities[:, column_map[0]],
        probabilities[:, column_map[1]],
    ])
    return mapped, decision


def predict_single(features: dict, model_name: str, input_method: str) -> dict:
    """Inference satu pasien menggunakan preprocessing contract dan payload model aktual."""
    try:
        normalized_input = _normalize_input_method(input_method)
        loaded = load_model(model_name)
        transformed = preprocess_single(model_name, features)
        model = loaded["model"]

        try:
            hard_prediction = np.asarray(model.predict(transformed), dtype=int).reshape(-1)
        except Exception as exc:
            raise InferenceError("predict() gagal dijalankan pada artifact aktual.") from exc

        if hard_prediction.shape[0] != 1:
            raise InferenceError("predict() tidak menghasilkan satu hard decision untuk satu pasien.")
        if not np.isin(hard_prediction, [0, 1]).all():
            raise InferenceError("predict() mengembalikan label selain 0 atau 1.")

        probabilities, decision = _probability_result(model, transformed)
        probability_benign = float(probabilities[0, 0])
        probability_malignant = float(probabilities[0, 1])
        hard_label = int(hard_prediction[0])

        response = {
            "status": "success",
            "model": model_name,
            "classification": hard_label,
            "classification_label": "Malignant" if hard_label == 1 else "Benign",
            "probability_benign": probability_benign,
            "probability_malignant": probability_malignant,
            "positive_class": "Malignant",
            "input_method": normalized_input,
            "model_status": cek_status_model(model_name).value,
        }
        if decision is not None:
            response["decision_score"] = float(decision[0])
        token = str(uuid.uuid4())
        response["provenance_token"] = token
        _trusted_single_responses[token] = response.copy()
        return response
    except (ModelLoadError, PreprocessingContractError, InferenceError) as exc:
        return {
            "status": "error",
            "model": model_name,
            "classification": None,
            "classification_label": None,
            "probability_benign": None,
            "probability_malignant": None,
            "positive_class": "Malignant",
            "input_method": str(input_method),
            "model_status": "MODEL_NOT_VERIFIED",
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "status": "error",
            "model": model_name,
            "classification": None,
            "classification_label": None,
            "probability_benign": None,
            "probability_malignant": None,
            "positive_class": "Malignant",
            "input_method": str(input_method),
            "model_status": "MODEL_NOT_VERIFIED",
            "error": f"Backend exception: {exc}",
        }


def predict_batch(dataframe: pd.DataFrame, model_name: str, input_method: str = "csv") -> dict:
    """Inference batch tanpa menghapus atau mengubah baris input yang invalid."""
    try:
        normalized_input = _normalize_input_method(input_method)
    except InferenceError as exc:
        normalized_input = str(input_method)
        return {
            "status": "error",
            "model": model_name,
            "jumlah_baris": 0,
            "jumlah_berhasil": 0,
            "jumlah_gagal": 0,
            "detail": [],
            "positive_class": "Malignant",
            "model_status": "MODEL_NOT_VERIFIED",
            "error": str(exc),
        }

    if not isinstance(dataframe, pd.DataFrame):
        return {
            "status": "error",
            "model": model_name,
            "jumlah_baris": 0,
            "jumlah_berhasil": 0,
            "jumlah_gagal": 0,
            "detail": [],
            "positive_class": "Malignant",
            "model_status": "MODEL_NOT_VERIFIED",
            "error": "Batch input harus berupa pandas DataFrame.",
        }

    total_rows = len(dataframe)
    if dataframe.empty:
        return {
            "status": "error",
            "model": model_name,
            "jumlah_baris": 0,
            "jumlah_berhasil": 0,
            "jumlah_gagal": 0,
            "detail": [],
            "positive_class": "Malignant",
            "model_status": "MODEL_NOT_VERIFIED",
            "error": "DataFrame batch kosong.",
        }

    try:
        loaded = load_model(model_name)
    except (ModelLoadError, PreprocessingContractError) as exc:
        details = []
        for position, source_index in enumerate(dataframe.index, start=1):
            details.append({
                "row_number": position,
                "source_index": _serialize_source_index(source_index),
                "status": "error",
                "classification": None,
                "classification_label": None,
                "probability_benign": None,
                "probability_malignant": None,
                "decision_score": None,
                "error": str(exc),
            })
        return {
            "status": "error",
            "model": model_name,
            "jumlah_baris": total_rows,
            "jumlah_berhasil": 0,
            "jumlah_gagal": total_rows,
            "detail": details,
            "positive_class": "Malignant",
            "input_method": normalized_input,
            "model_status": "MODEL_NOT_VERIFIED",
            "error": str(exc),
        }

    normalized_names = [_normalisasi_nama_kolom(nama) for nama in dataframe.columns]
    if len(normalized_names) != len(set(normalized_names)):
        duplicates = sorted({name for name in normalized_names if normalized_names.count(name) > 1})
        message = "Nama kolom duplikat setelah normalisasi: " + ", ".join(duplicates)
        details = [{
            "row_number": position,
            "source_index": _serialize_source_index(source_index),
            "status": "error",
            "classification": None,
            "classification_label": None,
            "probability_benign": None,
            "probability_malignant": None,
            "decision_score": None,
            "error": message,
        } for position, source_index in enumerate(dataframe.index, start=1)]
        return {
            "status": "error",
            "model": model_name,
            "jumlah_baris": total_rows,
            "jumlah_berhasil": 0,
            "jumlah_gagal": total_rows,
            "detail": details,
            "positive_class": "Malignant",
            "input_method": normalized_input,
            "model_status": "MODEL_NOT_VERIFIED",
            "error": message,
        }

    df = dataframe.copy()
    df.columns = normalized_names
    missing = [name for name in FITUR if name not in df.columns]
    extra = [name for name in df.columns if name not in FITUR]
    if missing or extra:
        message_parts = []
        if missing:
            message_parts.append("Fitur wajib hilang: " + ", ".join(missing))
        if extra:
            message_parts.append("Fitur tambahan tidak diperbolehkan: " + ", ".join(extra))
        message = "; ".join(message_parts)
        details = [{
            "row_number": position,
            "source_index": _serialize_source_index(source_index),
            "status": "error",
            "classification": None,
            "classification_label": None,
            "probability_benign": None,
            "probability_malignant": None,
            "decision_score": None,
            "error": message,
        } for position, source_index in enumerate(dataframe.index, start=1)]
        return {
            "status": "error",
            "model": model_name,
            "jumlah_baris": total_rows,
            "jumlah_berhasil": 0,
            "jumlah_gagal": total_rows,
            "detail": details,
            "positive_class": "Malignant",
            "input_method": normalized_input,
            "model_status": "MODEL_NOT_VERIFIED",
            "error": message,
        }

    valid_positions = []
    valid_rows = []
    invalid_by_position = {}

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        features = {name: row[name] for name in FITUR}
        try:
            preprocess_single(model_name, features)
        except PreprocessingContractError as exc:
            invalid_by_position[position] = str(exc)
            continue
        valid_positions.append(position)
        valid_rows.append({name: row[name] for name in FITUR})

    details = []
    valid_detail_map = {}

    if valid_rows:
        valid_dataframe = pd.DataFrame(valid_rows, columns=FITUR)
        try:
            transformed = preprocess_batch(model_name, valid_dataframe)
            model = loaded["model"]
            hard_prediction = np.asarray(model.predict(transformed), dtype=int).reshape(-1)
            if hard_prediction.shape[0] != len(valid_rows):
                raise InferenceError("predict() mengembalikan jumlah baris yang tidak sesuai batch.")
            if not np.isin(hard_prediction, [0, 1]).all():
                raise InferenceError("predict() mengembalikan label selain 0 atau 1.")
            probabilities, decision = _probability_result(model, transformed)

            for offset, position in enumerate(valid_positions):
                hard_label = int(hard_prediction[offset])
                detail = {
                    "row_number": position,
                    "source_index": _serialize_source_index(dataframe.index[position - 1]),
                    "status": "success",
                    "classification": hard_label,
                    "classification_label": "Malignant" if hard_label == 1 else "Benign",
                    "probability_benign": float(probabilities[offset, 0]),
                    "probability_malignant": float(probabilities[offset, 1]),
                    "decision_score": None,
                    "error": None,
                }
                if decision is not None:
                    detail["decision_score"] = float(decision[offset])
                valid_detail_map[position] = detail
        except (PreprocessingContractError, InferenceError, Exception) as exc:
            message = str(exc) if str(exc) else "Batch inference gagal dijalankan."
            for position in valid_positions:
                invalid_by_position[position] = message

    for position, source_index in enumerate(dataframe.index, start=1):
        if position in valid_detail_map:
            details.append(valid_detail_map[position])
        elif position in invalid_by_position:
            details.append({
                "row_number": position,
                "source_index": _serialize_source_index(source_index),
                "status": "error",
                "classification": None,
                "classification_label": None,
                "probability_benign": None,
                "probability_malignant": None,
                "decision_score": None,
                "error": invalid_by_position[position],
            })
        else:
            details.append({
                "row_number": position,
                "source_index": _serialize_source_index(source_index),
                "status": "error",
                "classification": None,
                "classification_label": None,
                "probability_benign": None,
                "probability_malignant": None,
                "decision_score": None,
                "error": "Baris tidak diproses karena batch inference gagal.",
            })

    successful = sum(detail["status"] == "success" for detail in details)
    failed = total_rows - successful
    if successful == total_rows:
        status = "success"
    elif successful == 0:
        status = "error"
    else:
        status = "partial_success"

    response = {
        "status": status,
        "model": model_name,
        "jumlah_baris": total_rows,
        "jumlah_berhasil": successful,
        "jumlah_gagal": failed,
        "detail": details,
        "positive_class": "Malignant",
        "input_method": normalized_input,
        "model_status": cek_status_model(model_name).value,
    }
    if failed:
        response["error"] = f"{failed} dari {total_rows} baris gagal diproses."
    token = str(uuid.uuid4())
    response["provenance_token"] = token
    _trusted_batch_responses[token] = response.copy()
    return response


def cek_status_model(nama_model: str) -> BackendStatus:
    path = MODEL_PATHS.get(nama_model)
    if path is None:
        return BackendStatus.PREDICTION_ERROR
    if not path.exists():
        return BackendStatus.MODEL_NOT_FOUND

    if not VERIFICATION_PATH.exists():
        return BackendStatus.MODEL_NOT_VERIFIED

    valid, _ = load_and_validate_marker(
        path=VERIFICATION_PATH,
        root=ROOT,
        model_paths=MODEL_PATHS,
        manifest_path=PREPROCESSING_MANIFEST_PATH,
    )
    if valid:
        try:
            load_model(nama_model)
            return BackendStatus.READY
        except (ModelLoadError, PreprocessingContractError):
            return BackendStatus.MODEL_NOT_VERIFIED
    return BackendStatus.MODEL_NOT_VERIFIED


def buat_respons_status(status: BackendStatus, nama_model: str) -> dict:
    pesan = {
        BackendStatus.MODEL_NOT_FOUND:
            f"Artefak {nama_model} belum tersedia. Prediksi tidak dijalankan.",
        BackendStatus.MODEL_NOT_VERIFIED:
            f"Artefak {nama_model} ditemukan, tetapi backend dan preprocessing inference belum terverifikasi.",
        BackendStatus.PREDICTION_ERROR:
            "Permintaan prediksi tidak valid atau gagal diproses.",
    }
    return {"status": status, "model": nama_model, "pesan": pesan.get(status, "Status backend tidak diketahui.")}


def prediksi_pasien(nama_model: str, fitur_berurut: dict, input_method: str = "manual") -> dict:
    """Wrapper backward-compatible untuk UI lama; menjalankan inference aktual secara langsung."""
    return predict_single(fitur_berurut, nama_model, input_method)


def simpan_prediksi_terverifikasi(respons: dict, fitur: dict, input_method: str, db_path=None, actual_label=None,
                                  id_pasien: int | None = None) -> int:
    """Menyimpan hanya respons inferensi nyata yang masih tercatat oleh backend."""
    if not isinstance(respons, dict):
        raise PermissionError("Respons prediksi tidak valid.")
    token = respons.get("provenance_token")
    trusted = _trusted_single_responses.get(token) if token else None
    if trusted is None or trusted != respons:
        raise PermissionError("Respons prediksi buatan atau sudah tidak sah.")
    if respons.get("status") != "success":
        raise PermissionError("Respons prediksi tidak sukses; hasil tidak disimpan.")

    nama_model = respons.get("model")
    if cek_status_model(nama_model) != BackendStatus.READY:
        raise PermissionError("Backend belum READY; hasil prediksi tidak disimpan.")

    classification = respons.get("classification_label")
    if classification not in {"Benign", "Malignant"}:
        raise ValueError("Respons prediksi tidak memuat classification yang sah.")
    probability_benign = respons.get("probability_benign")
    probability_malignant = respons.get("probability_malignant")
    if probability_benign is None or probability_malignant is None:
        raise ValueError("Respons prediksi tidak memuat probability yang sah.")
    if not isinstance(fitur, dict) or len(fitur) != 30:
        raise ValueError("Sebanyak 30 fitur harus disimpan sebagai JSON.")

    from basis_data.database import simpan_prediksi

    kwargs = {
        "model": nama_model,
        "classification": classification,
        "probability_benign": float(probability_benign),
        "probability_malignant": float(probability_malignant),
        "input_method": input_method,
        "fitur": fitur,
        "actual_label": actual_label,
        "verified": True,
        "id_pasien": id_pasien,
    }
    if db_path is not None:
        kwargs["db_path"] = db_path
    return simpan_prediksi(**kwargs)


def simpan_batch_prediksi_terverifikasi(respons: dict, dataframe: pd.DataFrame, db_path=None,
                                        label_asli: list | None = None, id_pasien: list | None = None) -> list[int]:
    """Menyimpan hanya baris sukses dari respons batch nyata yang masih tercatat backend."""
    if not isinstance(respons, dict) or not isinstance(dataframe, pd.DataFrame):
        raise PermissionError("Respons batch atau DataFrame tidak valid.")
    token = respons.get("provenance_token")
    trusted = _trusted_batch_responses.get(token) if token else None
    if trusted is None or trusted != respons:
        raise PermissionError("Respons batch buatan atau sudah tidak sah.")

    nama_model = respons.get("model")
    if cek_status_model(nama_model) != BackendStatus.READY:
        raise PermissionError("Backend belum READY; hasil batch tidak disimpan.")

    df = dataframe.copy()
    normalized_names = [_normalisasi_nama_kolom(name) for name in df.columns]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("Nama kolom batch duplikat.")
    df.columns = normalized_names

    missing = [name for name in FITUR if name not in df.columns]
    extra = [name for name in df.columns if name not in FITUR]
    if missing or extra:
        raise ValueError("Batch tidak memiliki kontrak 30 fitur canonical.")

    record_ids = []
    from basis_data.database import simpan_prediksi
    for detail in respons.get("detail", []):
        if detail.get("status") != "success":
            continue
        position = int(detail["row_number"])
        row = df.iloc[position - 1]
        fitur = {name: float(row[name]) for name in FITUR}
        kwargs = {
            "model": nama_model,
            "classification": detail["classification_label"],
            "probability_benign": float(detail["probability_benign"]),
            "probability_malignant": float(detail["probability_malignant"]),
            "input_method": "csv",
            "fitur": fitur,
            "actual_label": label_asli[position - 1] if label_asli and position - 1 < len(label_asli) else None,
            "verified": True,
            "id_pasien": id_pasien[position - 1] if id_pasien and position - 1 < len(id_pasien) else None,
        }
        if db_path is not None:
            kwargs["db_path"] = db_path
        record_ids.append(simpan_prediksi(**kwargs))
    return record_ids


_trusted_single_responses: dict[str, dict] = {}
_trusted_batch_responses: dict[str, dict] = {}
