import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basis_data.database import (
    ambil_detail,
    baca_riwayat,
    inisialisasi_database,
    simpan_prediksi,
)
from preprocessing.data import FITUR, validasi_csv
from utils import backend_adapter


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast_cancer.csv"


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def sample_features(dataset):
    row = dataset.iloc[99]
    return {name: float(row[name]) for name in FITUR}


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
@pytest.mark.parametrize("input_method", ["manual", "sample", "csv"])
def test_six_main_input_scenarios(model_name, input_method, sample_features):
    result = backend_adapter.predict_single(sample_features, model_name, input_method)

    assert result["status"] == "success"
    assert result["model"] == model_name
    assert result["input_method"] == input_method
    assert result["model_status"] == backend_adapter.cek_status_model(model_name).value
    assert result["classification"] in (0, 1)
    assert result["classification_label"] in ("Benign", "Malignant")
    assert np.isclose(
        result["probability_benign"] + result["probability_malignant"],
        1.0,
        rtol=1e-7,
        atol=1e-8,
    )


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_csv_batch_all_valid(model_name, dataset):
    csv_dataframe = dataset[FITUR].iloc[[0, 99, 200, 300, 568]].copy()

    result = backend_adapter.predict_batch(csv_dataframe, model_name, "csv")

    assert result["status"] == "success"
    assert result["jumlah_baris"] == 5
    assert result["jumlah_berhasil"] == 5
    assert result["jumlah_gagal"] == 0
    assert [detail["source_index"] for detail in result["detail"]] == [0, 99, 200, 300, 568]

    for detail in result["detail"]:
        assert np.isclose(
            detail["probability_benign"] + detail["probability_malignant"],
            1.0,
            rtol=1e-7,
            atol=1e-8,
        )


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_csv_batch_partial_invalid_preserves_source_index(model_name, dataset):
    csv_dataframe = dataset[FITUR].iloc[[10, 11, 12, 13]].copy()
    csv_dataframe.iloc[2, 0] = np.nan

    result = backend_adapter.predict_batch(csv_dataframe, model_name, "csv")

    assert result["status"] == "partial_success"
    assert result["jumlah_baris"] == 4
    assert result["jumlah_berhasil"] == 3
    assert result["jumlah_gagal"] == 1
    assert result["detail"][2]["row_number"] == 3
    assert result["detail"][2]["source_index"] == 12
    assert result["detail"][2]["status"] == "error"
    assert "Missing value" in result["detail"][2]["error"]


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_csv_batch_missing_feature(model_name, dataset):
    csv_dataframe = dataset[FITUR[:-1]].iloc[:3].copy()

    result = backend_adapter.predict_batch(csv_dataframe, model_name, "csv")

    assert result["status"] == "error"
    assert result["jumlah_gagal"] == 3
    assert "Fitur wajib hilang" in result["error"]


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_csv_batch_extra_feature(model_name, dataset):
    csv_dataframe = dataset[FITUR].iloc[:3].copy()
    csv_dataframe["extra_feature"] = 1.0

    result = backend_adapter.predict_batch(csv_dataframe, model_name, "csv")

    assert result["status"] == "error"
    assert result["jumlah_gagal"] == 3
    assert "Fitur tambahan" in result["error"]


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_csv_batch_duplicate_normalized_column(model_name, dataset):
    csv_dataframe = dataset[FITUR].iloc[:3].copy()
    csv_dataframe.columns = list(FITUR[:-1]) + ["radius_mean"]

    result = backend_adapter.predict_batch(csv_dataframe, model_name, "csv")

    assert result["status"] == "error"
    assert result["jumlah_gagal"] == 3
    assert "duplikat" in result["error"]


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_csv_batch_missing_and_non_numeric_values(model_name, dataset):
    missing_dataframe = dataset[FITUR].iloc[:3].copy()
    missing_dataframe.iloc[0, 0] = np.nan
    missing_result = backend_adapter.predict_batch(missing_dataframe, model_name, "csv")
    assert missing_result["status"] == "partial_success"
    assert "Missing value" in missing_result["detail"][0]["error"]

    invalid_dataframe = dataset[FITUR].iloc[:3].copy().astype(object)
    invalid_dataframe.iloc[0, 0] = "bukan_numerik"
    invalid_result = backend_adapter.predict_batch(invalid_dataframe, model_name, "csv")
    assert invalid_result["status"] == "partial_success"
    assert "angka numerik" in invalid_result["detail"][0]["error"]


def test_validasi_csv_all_valid_and_invalid_forms(dataset):
    valid_dataframe = dataset[FITUR].iloc[:4].copy()
    valid, valid_errors = validasi_csv(valid_dataframe)
    assert valid_errors == []
    assert valid is not None
    assert list(valid.columns) == FITUR

    missing_column = dataset[FITUR[:-1]].iloc[:4].copy()
    _, missing_errors = validasi_csv(missing_column)
    assert missing_errors

    extra_column = dataset[FITUR].iloc[:4].copy()
    extra_column["diagnosis"] = "M"
    valid_extra, extra_errors = validasi_csv(extra_column)
    assert extra_errors == []
    assert valid_extra is not None


@pytest.mark.parametrize("model_name", ["Random Forest", "Support Vector Machine"])
def test_single_vs_batch_consistency(model_name, dataset):
    dataframe = dataset[FITUR].iloc[[99, 200, 568]].copy()

    batch = backend_adapter.predict_batch(dataframe, model_name, "csv")

    assert batch["status"] == "success"
    for offset, source_index in enumerate([99, 200, 568]):
        features = {name: float(dataset.iloc[source_index][name]) for name in FITUR}
        single = backend_adapter.predict_single(features, model_name, "csv")
        detail = batch["detail"][offset]

        assert single["status"] == "success"
        assert detail["classification"] == single["classification"]
        assert np.isclose(detail["probability_benign"], single["probability_benign"])
        assert np.isclose(detail["probability_malignant"], single["probability_malignant"])


def test_svm_hard_decision_is_not_forced_to_probability_threshold(sample_features):
    result = backend_adapter.predict_single(sample_features, "Support Vector Machine", "sample")

    assert result["status"] == "success"
    assert result["classification"] == 0
    assert result["probability_malignant"] > 0.5
    assert "decision_score" in result


def test_invalid_model_name_and_input_method(sample_features):
    invalid_model = backend_adapter.predict_single(sample_features, "Unknown Model", "manual")
    assert invalid_model["status"] == "error"
    assert invalid_model["model_status"] == "MODEL_NOT_VERIFIED"

    invalid_method = backend_adapter.predict_single(sample_features, "Random Forest", "unknown")
    assert invalid_method["status"] == "error"
    assert invalid_method["error"]


def test_missing_model_and_backend_exception(monkeypatch, sample_features):
    original_paths = backend_adapter.MODEL_PATHS.copy()

    try:
        missing_path = ROOT / "models" / "model_that_does_not_exist.pkl"
        monkeypatch.setitem(backend_adapter.MODEL_PATHS, "Random Forest", missing_path)
        missing_result = backend_adapter.predict_single(sample_features, "Random Forest", "manual")
        assert missing_result["status"] == "error"
        assert "tidak ditemukan" in missing_result["error"]

        monkeypatch.setitem(backend_adapter.MODEL_PATHS, "Random Forest", original_paths["Random Forest"])
        original_preprocess = backend_adapter.preprocess_single

        def backend_exception(*args, **kwargs):
            raise RuntimeError("backend exception simulated")

        monkeypatch.setattr(backend_adapter, "preprocess_single", backend_exception)
        exception_result = backend_adapter.predict_single(sample_features, "Random Forest", "manual")

        monkeypatch.setattr(backend_adapter, "preprocess_single", original_preprocess)
        assert exception_result["status"] == "error"
        assert "Backend exception" in exception_result["error"]
    finally:
        monkeypatch.setattr(backend_adapter, "preprocess_single", backend_adapter.preprocess_single)


def test_probability_invalid_handling(monkeypatch, sample_features):
    loaded = backend_adapter.load_model("Random Forest")
    model = loaded["model"]

    original_predict_proba = model.predict_proba

    def invalid_probability(_):
        return np.asarray([[0.2, 0.2]])

    monkeypatch.setattr(model, "predict_proba", invalid_probability)

    transformed = backend_adapter.preprocess_single("Random Forest", sample_features)
    with pytest.raises(backend_adapter.InferenceError, match="tidak sama dengan 1"):
        backend_adapter._probability_result(model, transformed)

    monkeypatch.setattr(model, "predict_proba", original_predict_proba)


def test_database_integration_uses_temporary_database_and_blocks_unverified_save():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "validation_history.db"

        with pytest.raises(PermissionError):
            simpan_prediksi(
                model="Random Forest",
                classification="Malignant",
                probability_benign=0.2,
                probability_malignant=0.8,
                input_method="manual",
                fitur={name: 1.0 for name in FITUR},
                db_path=db_path,
                verified=False,
            )

        db = inisialisasi_database(db_path)
        history_before = baca_riwayat(db_path=db)
        assert history_before.empty

        record_id = simpan_prediksi(
            model="Random Forest",
            classification="Malignant",
            probability_benign=0.2,
            probability_malignant=0.8,
            input_method="manual",
            fitur={name: 1.0 for name in FITUR},
            db_path=db,
            verified=True,
        )

        history_after = baca_riwayat(db_path=db)
        detail = ambil_detail(record_id, db_path=db)

        assert len(history_after) == 1
        assert detail is not None
        assert detail["Hasil klasifikasi"] == "Malignant"
        assert len(detail["Fitur"]) == 30


def test_response_contract_single_and_batch(sample_features):
    single = backend_adapter.predict_single(sample_features, "Random Forest", "manual")
    assert set([
        "status", "model", "classification", "classification_label",
        "probability_benign", "probability_malignant", "positive_class",
        "input_method", "model_status",
    ]).issubset(single.keys())

    batch = backend_adapter.predict_batch(
        pd.DataFrame([sample_features], columns=FITUR),
        "Random Forest",
        "csv",
    )
    assert set([
        "status", "model", "jumlah_baris", "jumlah_berhasil",
        "jumlah_gagal", "detail", "positive_class", "input_method", "model_status",
    ]).issubset(batch.keys())
