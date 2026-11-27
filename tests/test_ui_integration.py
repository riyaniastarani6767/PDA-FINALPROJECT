from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from preprocessing.data import FITUR
from utils import backend_adapter
from utils.ui_storage import jalankan_penyimpanan_sekali


ROOT = Path(__file__).resolve().parents[1]


def test_ui_storage_duplicate_prevention():
    class State:
        def __init__(self):
            self.data = {}
        def setdefault(self, key, default):
            return self.data.setdefault(key, default)
        def __setitem__(self, key, value):
            self.data[key] = value

    state = State()
    calls = {"count": 0}

    def save():
        calls["count"] += 1
        return [123]

    first = jalankan_penyimpanan_sekali(state, "single:token", save)
    second = jalankan_penyimpanan_sekali(state, "single:token", save)

    assert first["status"] == "saved"
    assert second["status"] == "already_saved"
    assert calls["count"] == 1


def test_fabricated_single_response_rejected(tmp_path):
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    features = {name: float(dataset.iloc[0][name]) for name in FITUR}
    fabricated = {
        "status": "success",
        "model": "Random Forest",
        "classification": 1,
        "classification_label": "Malignant",
        "probability_benign": 0.2,
        "probability_malignant": 0.8,
        "positive_class": "Malignant",
        "input_method": "manual",
        "model_status": "READY",
    }
    try:
        backend_adapter.simpan_prediksi_terverifikasi(
            fabricated, features, "manual", db_path=tmp_path / "history.db"
        )
        assert False, "Respons buatan harus ditolak"
    except PermissionError:
        pass


def test_trusted_batch_storage_uses_temporary_database(tmp_path):
    dataset = pd.read_csv(ROOT / "data/breast_cancer.csv")
    dataframe = dataset[FITUR].iloc[[0, 99, 200]].copy()
    response = backend_adapter.predict_batch(dataframe, "Random Forest", "csv")

    record_ids = backend_adapter.simpan_batch_prediksi_terverifikasi(
        response, dataframe, db_path=tmp_path / "history.db"
    )

    assert len(record_ids) == 3


def test_prediction_page_starts_with_verified_backend():
    app = AppTest.from_file(str(ROOT / "pages/02_Prediksi.py"), default_timeout=20)
    app.run()
    assert not app.exception
    assert len(app.success) > 0 or len(app.info) > 0 or len(app.warning) > 0


def test_history_page_starts():
    app = AppTest.from_file(str(ROOT / "pages/03_Riwayat.py"), default_timeout=20)
    app.run()
    assert not app.exception
