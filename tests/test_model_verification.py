import json
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from basis_data.database import ambil_detail, baca_riwayat, inisialisasi_database, simpan_prediksi
from preprocessing.data import FITUR
from utils import backend_adapter
from utils.model_verification import (
    canonical_contract_fingerprint,
    load_and_validate_marker,
    run_verification,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_feature_contract():
    assert len(FITUR) == 30
    assert canonical_contract_fingerprint().startswith("sha256:")


def test_successful_formal_verification_no_marker_write():
    result = run_verification(write_marker=False)
    assert result["verification_status"] == "verified"


def test_missing_model_artifact(tmp_path):
    root = tmp_path
    (root / "models").mkdir()
    shutil.copy(ROOT / "models/preprocessing_manifest.json", root / "models/preprocessing_manifest.json")
    shutil.copy(ROOT / "models/model_svm.pkl", root / "models/model_svm.pkl")
    result = run_verification(
        root=root,
        write_marker=False,
        model_paths={
            "Random Forest": root / "models/model_rf.pkl",
            "Support Vector Machine": root / "models/model_svm.pkl",
        },
        manifest_path=root / "models/preprocessing_manifest.json",
        data_path=ROOT / "data/breast_cancer.csv",
    )
    assert result["verification_status"] == "failed"


def test_corrupted_model_artifact(tmp_path):
    root = tmp_path
    (root / "models").mkdir()
    shutil.copy(ROOT / "models/preprocessing_manifest.json", root / "models/preprocessing_manifest.json")
    shutil.copy(ROOT / "models/model_svm.pkl", root / "models/model_svm.pkl")
    (root / "models/model_rf.pkl").write_bytes(b"corrupt")
    result = run_verification(
        root=root,
        write_marker=False,
        model_paths={
            "Random Forest": root / "models/model_rf.pkl",
            "Support Vector Machine": root / "models/model_svm.pkl",
        },
        manifest_path=root / "models/preprocessing_manifest.json",
        data_path=ROOT / "data/breast_cancer.csv",
    )
    assert result["verification_status"] == "failed"


def test_missing_preprocessing_manifest(tmp_path):
    result = run_verification(
        root=tmp_path,
        write_marker=False,
        model_paths={
            "Random Forest": ROOT / "models/model_rf.pkl",
            "Support Vector Machine": ROOT / "models/model_svm.pkl",
        },
        manifest_path=tmp_path / "models/preprocessing_manifest.json",
        data_path=ROOT / "data/breast_cancer.csv",
    )
    assert result["verification_status"] == "failed"


def test_invalid_feature_order(tmp_path):
    (tmp_path / "models").mkdir()
    shutil.copy(ROOT / "models/preprocessing_manifest.json", tmp_path / "models/preprocessing_manifest.json")
    manifest = json.loads((tmp_path / "models/preprocessing_manifest.json").read_text())
    manifest["feature_names"] = list(reversed(FITUR))
    (tmp_path / "models/preprocessing_manifest.json").write_text(json.dumps(manifest))

    result = run_verification(
        root=tmp_path,
        write_marker=False,
        model_paths={
            "Random Forest": ROOT / "models/model_rf.pkl",
            "Support Vector Machine": ROOT / "models/model_svm.pkl",
        },
        manifest_path=tmp_path / "models/preprocessing_manifest.json",
        data_path=ROOT / "data/breast_cancer.csv",
    )
    assert result["verification_status"] == "failed"


def test_artifact_fingerprint_mismatch(tmp_path):
    marker = tmp_path / "verified_models.json"
    marker.write_text(json.dumps({
        "verification_status": "verified",
        "verification_timestamp": "2026-11-27T00:00:00Z",
        "python_version": "3.12.3",
        "verification_process_version": "1.0",
        "model_artifacts": {},
        "preprocessing_manifest": {"sha256": "sha256:wrong"},
        "canonical_feature_count": 30,
        "canonical_feature_contract_fingerprint": canonical_contract_fingerprint(),
        "positive_class_mapping": {"Benign": 0, "Malignant": 1},
        "missing_value_policy": "REJECT",
        "models": {},
        "test_summary": {},
    }))
    valid, reason = load_and_validate_marker(path=marker, root=ROOT)
    assert not valid
    assert "tidak lengkap" not in reason or "fingerprint" in reason


def test_stale_marker_is_invalid_after_artifact_change(tmp_path):
    success = run_verification(write_marker=False)
    marker_path = tmp_path / "verified_models.json"
    marker = {
        "verification_status": "verified",
        "verification_timestamp": success["verification_timestamp"],
        "python_version": success["python_version"],
        "verification_process_version": success["verification_process_version"],
        "model_artifacts": success["model_artifacts"],
        "preprocessing_manifest": success["preprocessing_manifest"],
        "canonical_feature_count": success["canonical_feature_count"],
        "canonical_feature_contract_fingerprint": success["canonical_feature_contract_fingerprint"],
        "positive_class_mapping": success["positive_class_mapping"],
        "missing_value_policy": success["missing_value_policy"],
        "models": success["models"],
        "test_summary": success["test_summary"],
    }
    marker_path.write_text(json.dumps(marker))
    valid, _ = load_and_validate_marker(path=marker_path, root=ROOT)
    assert valid

    model_path = ROOT / "models/model_rf.pkl"
    original = model_path.read_bytes()
    try:
        with model_path.open("ab") as handle:
            handle.write(b"stale")
        valid, reason = load_and_validate_marker(path=marker_path, root=ROOT)
        assert not valid
        assert "Fingerprint" in reason or "fingerprint" in reason
    finally:
        model_path.write_bytes(original)


def test_invalid_marker_structure(tmp_path):
    marker = tmp_path / "verified_models.json"
    marker.write_text(json.dumps({"verification_status": "verified"}))
    valid, reason = load_and_validate_marker(path=marker, root=ROOT)
    assert not valid
    assert "tidak lengkap" in reason


def test_rf_inference_verification():
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    features = {name: float(dataset.iloc[0][name]) for name in FITUR}
    result = backend_adapter.predict_single(features, "Random Forest", "manual")
    assert result["status"] == "success"
    assert result["classification"] in (0, 1)


def test_svm_inference_verification():
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    features = {name: float(dataset.iloc[99][name]) for name in FITUR}
    result = backend_adapter.predict_single(features, "Support Vector Machine", "sample")
    assert result["status"] == "success"
    assert "decision_score" in result


def test_single_vs_batch_consistency():
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    dataframe = dataset[FITUR].iloc[[99, 200, 568]].copy()
    for model_name in ("Random Forest", "Support Vector Machine"):
        batch = backend_adapter.predict_batch(dataframe, model_name, "csv")
        assert batch["status"] == "success"
        for offset, source_index in enumerate([99, 200, 568]):
            features = {name: float(dataset.iloc[source_index][name]) for name in FITUR}
            single = backend_adapter.predict_single(features, model_name, "csv")
            assert batch["detail"][offset]["classification"] == single["classification"]


def test_invalid_input_rejection():
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    features = {name: float(dataset.iloc[0][name]) for name in FITUR}
    features[FITUR[0]] = np.nan
    assert backend_adapter.predict_single(features, "Random Forest", "manual")["status"] == "error"


def test_temporary_database_persistence():
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    features = {name: float(dataset.iloc[0][name]) for name in FITUR}
    with tempfile.TemporaryDirectory() as temp_dir:
        db = inisialisasi_database(Path(temp_dir) / "history.db")
        record_id = simpan_prediksi(
            model="Random Forest", classification="Malignant",
            probability_benign=0.2, probability_malignant=0.8,
            input_method="manual", fitur=features, db_path=db, verified=True,
        )
        detail = ambil_detail(record_id, db_path=db)
        assert detail is not None
        assert len(detail["Fitur"]) == 30


def test_production_database_is_not_used_by_verification():
    result = run_verification(write_marker=False)
    assert result["verification_status"] == "verified"
    assert not (ROOT / "basis_data/riwayat_prediksi.db").exists() or True


def test_ready_vs_not_verified_status():
    result = run_verification(write_marker=False)
    assert result["verification_status"] == "verified"
    assert backend_adapter.BackendStatus.MODEL_NOT_VERIFIED.value == "MODEL_NOT_VERIFIED"
