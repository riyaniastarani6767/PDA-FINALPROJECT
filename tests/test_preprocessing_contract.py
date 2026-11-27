import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from preprocessing.data import (
    FITUR,
    ImputerMedianManual,
    StandardScalerManual,
    bagi_stratified,
    RANDOM_STATE,
)
from utils.backend_adapter import (
    PreprocessingContractError,
    preprocess_batch,
    preprocess_single,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "preprocessing_manifest.json"
DATA_PATH = ROOT / "data" / "breast_cancer.csv"


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def single_row(dataset):
    return dataset.iloc[0]


def test_manifest_is_valid_and_matches_canonical_features(manifest):
    assert manifest["manifest_version"] == "1.0"
    assert manifest["feature_count"] == 30
    assert manifest["feature_names"] == FITUR
    assert manifest["missing_value_policy"] == "reject"
    assert manifest["negative_value_policy"] == "reject"


def test_manifest_target_mapping_and_positive_class(manifest):
    assert manifest["target_mapping"] == {"B": 0, "M": 1}
    assert manifest["positive_class"] == "Malignant"


def test_model_artifacts_exist_and_scalers_load():
    rf = joblib.load(ROOT / "models" / "model_rf.pkl")
    svm = joblib.load(ROOT / "models" / "model_svm.pkl")

    assert rf["fitur"] == FITUR
    assert svm["fitur"] == FITUR
    assert hasattr(rf["scaler"], "transform")
    assert hasattr(svm["scaler"], "transform")


def test_single_input_complete_uses_canonical_order(single_row):
    features = {name: float(single_row[name]) for name in FITUR}

    transformed = preprocess_single("Random Forest", features)

    assert transformed.shape == (1, 30)
    assert np.isfinite(transformed).all()


def test_single_input_with_different_dict_order(single_row):
    features = {name: float(single_row[name]) for name in reversed(FITUR)}

    transformed = preprocess_single("Random Forest", features)

    scaler = joblib.load(ROOT / "models" / "model_rf.pkl")["scaler"]
    canonical = np.asarray([float(single_row[name]) for name in FITUR], dtype=float).reshape(1, -1)
    expected = scaler.transform(canonical)

    assert np.allclose(transformed, expected)


def test_single_missing_feature_rejected(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features.pop(FITUR[0])

    with pytest.raises(PreprocessingContractError, match="Fitur wajib hilang"):
        preprocess_single("Random Forest", features)


def test_single_extra_feature_rejected(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features["extra_feature"] = 1.0

    with pytest.raises(PreprocessingContractError, match="Fitur tambahan"):
        preprocess_single("Random Forest", features)


def test_duplicate_normalized_feature_names_rejected_single_dict(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features["radius mean"] = features["radius_mean"]

    with pytest.raises(PreprocessingContractError, match="Nama kolom duplikat"):
        preprocess_single("Random Forest", features)


def test_non_numeric_value_rejected(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features[FITUR[0]] = "bukan_angka"

    with pytest.raises(PreprocessingContractError, match="angka numerik"):
        preprocess_single("Random Forest", features)


def test_missing_value_rejected(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features[FITUR[0]] = np.nan

    with pytest.raises(PreprocessingContractError, match="Missing value"):
        preprocess_single("Random Forest", features)


def test_negative_value_rejected(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features[FITUR[0]] = -1.0

    with pytest.raises(PreprocessingContractError, match="negatif"):
        preprocess_single("Random Forest", features)


def test_infinite_value_rejected(single_row):
    features = {name: float(single_row[name]) for name in FITUR}
    features[FITUR[0]] = np.inf

    with pytest.raises(PreprocessingContractError, match="tak-hingga"):
        preprocess_single("Random Forest", features)


def test_batch_missing_feature_rejected(dataset):
    batch = dataset[FITUR[:-1]].copy()

    with pytest.raises(PreprocessingContractError, match="Fitur wajib hilang"):
        preprocess_batch("Random Forest", batch)


def test_batch_extra_feature_rejected(dataset):
    batch = dataset[FITUR].copy()
    batch["extra_feature"] = 1.0

    with pytest.raises(PreprocessingContractError, match="Fitur tambahan"):
        preprocess_batch("Random Forest", batch)


def test_batch_duplicate_normalized_names_rejected(dataset):
    batch = dataset[FITUR].iloc[:5].copy()
    batch.columns = [*list(FITUR[:-1]), "radius_mean"]

    with pytest.raises(PreprocessingContractError, match="duplikat"):
        preprocess_batch("Random Forest", batch)


def test_batch_missing_value_rejected(dataset):
    batch = dataset[FITUR].iloc[:5].copy()
    batch.iloc[0, 0] = np.nan

    with pytest.raises(PreprocessingContractError, match="Missing value"):
        preprocess_batch("Random Forest", batch)


def test_batch_negative_value_rejected(dataset):
    batch = dataset[FITUR].iloc[:5].copy()
    batch.iloc[0, 0] = -1.0

    with pytest.raises(PreprocessingContractError, match="negatif"):
        preprocess_batch("Random Forest", batch)


def test_batch_and_single_use_same_scaler_transform(dataset):
    row = dataset.iloc[0]
    features = {name: float(row[name]) for name in FITUR}

    single = preprocess_single("Random Forest", features)
    batch = preprocess_batch("Random Forest", pd.DataFrame([features]))

    assert np.allclose(single, batch)


def test_batch_reordered_columns_match_canonical(dataset):
    batch = dataset[FITUR].iloc[:3].copy()
    reordered = batch[list(reversed(FITUR))]

    transformed_reordered = preprocess_batch("Random Forest", reordered)
    transformed_canonical = preprocess_batch("Random Forest", batch)

    assert np.allclose(transformed_reordered, transformed_canonical)


def test_training_inference_transform_equality_without_missing_value(dataset):
    y = (dataset["diagnosis"].astype(str).str.upper() == "M").astype(int).to_numpy()
    X = dataset[FITUR].astype(float)
    X_train, X_test, y_train, y_test = bagi_stratified(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    imputer = ImputerMedianManual().fit(X_train)
    training_scaler = StandardScalerManual().fit(imputer.transform(X_train))

    expected_test = training_scaler.transform(imputer.transform(X_test))

    rf_payload = joblib.load(ROOT / "models" / "model_rf.pkl")
    actual_test = rf_payload["scaler"].transform(
        np.asarray(imputer.transform(X_test), dtype=float)
    )

    assert np.allclose(expected_test, actual_test)


def test_scaler_is_not_refit_by_preprocessing(dataset):
    payload = joblib.load(ROOT / "models" / "model_rf.pkl")
    scaler = payload["scaler"]
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    batch = dataset[FITUR].iloc[:5].copy()
    preprocess_batch("Random Forest", batch)

    assert np.array_equal(scaler.mean_, mean_before)
    assert np.array_equal(scaler.scale_, scale_before)


def test_svm_and_rf_share_training_scaler():
    rf_payload = joblib.load(ROOT / "models" / "model_rf.pkl")
    svm_payload = joblib.load(ROOT / "models" / "model_svm.pkl")

    assert np.array_equal(rf_payload["scaler"].mean_, svm_payload["scaler"].mean_)
    assert np.array_equal(rf_payload["scaler"].scale_, svm_payload["scaler"].scale_)


def test_manifest_feature_names_equal_model_payloads(manifest):
    rf_payload = joblib.load(ROOT / "models" / "model_rf.pkl")
    svm_payload = joblib.load(ROOT / "models" / "model_svm.pkl")

    assert manifest["feature_names"] == rf_payload["fitur"]
    assert manifest["feature_names"] == svm_payload["fitur"]


def test_non_duplicate_batch_columns_are_allowed_if_all_features_exist(dataset):
    batch = dataset[FITUR].iloc[:3].copy()
    batch = batch[list(reversed(FITUR))]

    transformed = preprocess_batch("Random Forest", batch)

    assert transformed.shape == (3, 30)
