import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocessing.data import FITUR
from utils import backend_adapter


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast_cancer.csv"


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def deterministic_rows(dataset):
    return {
        "row_0": dataset.iloc[0],
        "svm_hard_probability_divergence": dataset.iloc[99],
    }


def test_load_random_forest():
    loaded = backend_adapter.load_model("Random Forest")
    assert loaded["model_name"] == "Random Forest"
    assert loaded["status"] == "loaded_not_verified"
    assert loaded["feature_names"] == FITUR


def test_load_svm():
    loaded = backend_adapter.load_model("Support Vector Machine")
    assert loaded["model_name"] == "Support Vector Machine"
    assert loaded["status"] == "loaded_not_verified"
    assert loaded["feature_names"] == FITUR


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_single_prediction_real_artifact(model_name, deterministic_rows):
    row = deterministic_rows["row_0"]
    features = {name: float(row[name]) for name in FITUR}

    result = backend_adapter.predict_single(features, model_name, "manual")

    assert result["status"] == "success"
    assert result["model_status"] == backend_adapter.cek_status_model(model_name).value
    assert result["classification"] in (0, 1)
    assert result["classification_label"] in ("Benign", "Malignant")
    assert 0.0 <= result["probability_benign"] <= 1.0
    assert 0.0 <= result["probability_malignant"] <= 1.0
    assert np.isclose(
        result["probability_benign"] + result["probability_malignant"],
        1.0,
        rtol=1e-7,
        atol=1e-8,
    )


def test_svm_decision_score_available(deterministic_rows):
    row = deterministic_rows["row_0"]
    features = {name: float(row[name]) for name in FITUR}
    result = backend_adapter.predict_single(features, "Support Vector Machine", "sample")

    assert result["status"] == "success"
    assert "decision_score" in result
    assert np.isfinite(result["decision_score"])


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_batch_prediction_real_artifact(model_name, dataset):
    dataframe = dataset[FITUR].iloc[:5].copy()
    result = backend_adapter.predict_batch(dataframe, model_name, "csv")

    assert result["status"] == "success"
    assert result["jumlah_baris"] == 5
    assert result["jumlah_berhasil"] == 5
    assert result["jumlah_gagal"] == 0
    assert len(result["detail"]) == 5

    for detail in result["detail"]:
        assert detail["status"] == "success"
        assert np.isclose(
            detail["probability_benign"] + detail["probability_malignant"],
            1.0,
            rtol=1e-7,
            atol=1e-8,
        )


def test_positive_class_mapping(deterministic_rows):
    row = deterministic_rows["row_0"]
    features = {name: float(row[name]) for name in FITUR}

    for model_name in ("Random Forest", "Support Vector Machine"):
        result = backend_adapter.predict_single(features, model_name, "manual")
        assert result["positive_class"] == "Malignant"
        expected_label = "Malignant" if result["classification"] == 1 else "Benign"
        assert result["classification_label"] == expected_label


def test_hard_prediction_is_not_replaced_by_probability_threshold(deterministic_rows):
    row = deterministic_rows["svm_hard_probability_divergence"]
    features = {name: float(row[name]) for name in FITUR}

    result = backend_adapter.predict_single(features, "Support Vector Machine", "sample")

    assert result["status"] == "success"
    assert result["classification"] == 0
    assert result["probability_malignant"] > 0.5


def test_feature_order(deterministic_rows):
    row = deterministic_rows["row_0"]
    reversed_features = {name: float(row[name]) for name in reversed(FITUR)}

    canonical = backend_adapter.predict_single(reversed_features, "Random Forest", "manual")

    assert canonical["status"] == "success"


def test_missing_value_rejection(deterministic_rows):
    row = deterministic_rows["row_0"]
    features = {name: float(row[name]) for name in FITUR}
    features[FITUR[0]] = np.nan

    result = backend_adapter.predict_single(features, "Random Forest", "manual")

    assert result["status"] == "error"
    assert "Missing value" in result["error"]


def test_invalid_model_name():
    result = backend_adapter.predict_single({}, "Unknown Model", "manual")
    assert result["status"] == "error"
    assert "Nama model tidak dikenali" in result["error"]


def test_missing_model_artifact(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_path = Path(temp_dir) / "missing.pkl"
        monkeypatch.setitem(backend_adapter.MODEL_PATHS, "Random Forest", missing_path)

        with pytest.raises(backend_adapter.ModelLoadError, match="tidak ditemukan"):
            backend_adapter.load_model("Random Forest")


def test_missing_model_artifact_load_error(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_path = Path(temp_dir) / "missing.pkl"
        monkeypatch.setitem(backend_adapter.MODEL_PATHS, "Random Forest", missing_path)

        with pytest.raises(backend_adapter.ModelLoadError, match="tidak ditemukan"):
            backend_adapter.load_model("Random Forest")


def test_corrupted_artifact_handling(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        corrupted_path = Path(temp_dir) / "corrupt.pkl"
        corrupted_path.write_bytes(b"not-a-joblib-model")
        monkeypatch.setitem(backend_adapter.MODEL_PATHS, "Random Forest", corrupted_path)

        with pytest.raises(backend_adapter.ModelLoadError, match="rusak"):
            backend_adapter.load_model("Random Forest")


def test_single_and_batch_preprocessing_consistency(dataset):
    row = dataset.iloc[0]
    features = {name: float(row[name]) for name in FITUR}

    single = backend_adapter.preprocess_single("Random Forest", features)
    batch = backend_adapter.preprocess_batch("Random Forest", pd.DataFrame([features]))

    assert np.allclose(single, batch)


@pytest.mark.parametrize("input_method", ["manual", "sample", "csv"])
def test_input_methods(input_method, deterministic_rows):
    row = deterministic_rows["row_0"]
    features = {name: float(row[name]) for name in FITUR}

    result = backend_adapter.predict_single(features, "Random Forest", input_method)

    assert result["status"] == "success"
    assert result["input_method"] in ("manual", "sample", "csv")


def test_batch_preserves_row_identity_and_reports_invalid_rows(dataset):
    dataframe = dataset[FITUR].iloc[:4].copy()
    dataframe = dataframe.copy()
    dataframe.iloc[1, 0] = np.nan

    result = backend_adapter.predict_batch(dataframe, "Random Forest", "csv")

    assert result["status"] == "partial_success"
    assert result["jumlah_baris"] == 4
    assert result["jumlah_berhasil"] == 3
    assert result["jumlah_gagal"] == 1
    assert result["detail"][1]["row_number"] == 2
    assert result["detail"][1]["status"] == "error"
