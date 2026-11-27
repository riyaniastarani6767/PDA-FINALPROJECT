from pathlib import Path

import pandas as pd
import streamlit as st

from components.empty_state import tampilkan_empty_state
from components.prediction import tampilkan_area_hasil
from interpretasi_shap import shap_svm
from interpretasi_shap.tab_shap import tampilkan_shap_lokal
from preprocessing.data import FITUR, label_fitur, validasi_csv, validasi_satu_pasien
from utils.backend_adapter import (
    BackendStatus,
    cek_status_model,
    prediksi_pasien,
    preprocess_single,
    predict_batch,
    simpan_batch_prediksi_terverifikasi,
    simpan_prediksi_terverifikasi,
)
from utils.data_loader import DatasetLoadError, muat_dataset_ui, tabel_daftar_fitur
from utils.ui_storage import jalankan_penyimpanan_sekali


ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "assets" / "styles.css"
st.set_page_config(page_title="Prediksi", page_icon="🧪", layout="wide")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def muat_data_prediksi():
    return muat_dataset_ui()


@st.cache_resource(show_spinner=False)
def muat_explainer_shap():
    """KernelExplainer SVM dari artefak training (model_svm.pkl + shap_background.pkl), tanpa training ulang."""
    import joblib

    payload = joblib.load(ROOT / "models" / "model_svm.pkl")
    background = joblib.load(ROOT / "models" / "shap_background.pkl")
    return shap_svm.buat_explainer(payload["model"], background)


def label_kategori(nama: str) -> str:
    if nama.endswith("_mean"):
        return "Mean"
    if nama.endswith("_se"):
        return "Standard Error"
    if nama.endswith("_worst"):
        return "Worst"
    return "Lainnya"


def format_status(status: BackendStatus) -> str:
    if status == BackendStatus.MODEL_NOT_FOUND:
        return "Belum tersedia"
    if status == BackendStatus.MODEL_NOT_VERIFIED:
        return "Artefak tersedia—belum terverifikasi"
    if status == BackendStatus.READY:
        return "Siap digunakan"
    return "Error"


st.title("Prediksi Tumor Payudara")
st.markdown(
    "Masukkan karakteristik inti sel untuk melihat hasil klasifikasi menggunakan model "
    "Random Forest atau Support Vector Machine."
)

st.markdown('<div class="section-heading">Pemilihan Model</div>', unsafe_allow_html=True)
nama_model = st.selectbox(
    "Model",
    options=["Random Forest", "Support Vector Machine"],
)
status_model = cek_status_model(nama_model)
if status_model == BackendStatus.MODEL_NOT_FOUND:
    st.warning(f"{nama_model}: {format_status(status_model)}")
elif status_model == BackendStatus.MODEL_NOT_VERIFIED:
    st.info(f"{nama_model}: {format_status(status_model)}")
elif status_model == BackendStatus.READY:
    st.success(f"{nama_model}: {format_status(status_model)}")
else:
    st.error(f"{nama_model}: {format_status(status_model)}")
st.caption(
    "Keberadaan artefak model belum berarti backend siap. Prediksi hanya dapat dijalankan "
    "setelah preprocessing inference dan urutan fitur diverifikasi."
)

try:
    df_dataset = muat_data_prediksi()
except DatasetLoadError as exc:
    df_dataset = None
    st.error(str(exc))

st.markdown('<div class="section-heading">Metode Input</div>', unsafe_allow_html=True)
metode_input = st.radio(
    "Pilih metode input",
    options=["Input Manual", "Pilih Sampel Dataset", "Upload CSV"],
    horizontal=True,
)

nilai_input: dict | None = None
csv_batch_dataframe: pd.DataFrame | None = None
actual_label_sample: str | None = None

if metode_input == "Input Manual":
    if "reset_manual" not in st.session_state:
        st.session_state.reset_manual = 0

    kolom_kiri, kolom_tengah, kolom_kanan = st.columns(3)
    kategori_kolom = {
        "Mean": kolom_kiri,
        "Standard Error": kolom_tengah,
        "Worst": kolom_kanan,
    }

    nilai_manual = {}
    tabel_fitur = tabel_daftar_fitur()
    for kategori, kolom in kategori_kolom.items():
        with kolom:
            st.markdown(f"**{kategori}**")
            daftar = tabel_fitur[tabel_fitur["Kategori"] == kategori]
            for _, baris in daftar.iterrows():
                nama = baris["Nama fitur asli"]
                kunci = f"manual_{st.session_state.reset_manual}_{nama}"
                nilai_manual[nama] = st.number_input(
                    label_fitur(nama),
                    min_value=0.0,
                    value=None,
                    step=0.001,
                    format="%.6g",
                    key=kunci,
                    help=f"Nilai numerik untuk kolom asli `{nama}`.",
                )

    tombol_reset, tombol_status = st.columns([1, 2])
    with tombol_reset:
        if st.button("Reset form", use_container_width=True):
            st.session_state.reset_manual += 1
            st.rerun()
    with tombol_status:
        error_manual = validasi_satu_pasien(nilai_manual)
        if error_manual:
            st.caption(" ".join(error_manual))

    nilai_input = nilai_manual

elif metode_input == "Pilih Sampel Dataset":
    if df_dataset is None:
        tampilkan_empty_state(
            "Sampel dataset tidak tersedia",
            "Data dataset tidak dapat dimuat. Tidak ada sampel yang dapat dipilih.",
            icon="📋",
        )
    else:
        opsi_id = df_dataset["id"].tolist() if "id" in df_dataset.columns else df_dataset.index.tolist()
        id_sampel = st.selectbox("ID sampel", options=opsi_id)

        if "id" in df_dataset.columns:
            baris = df_dataset.loc[df_dataset["id"] == id_sampel].head(1)
        else:
            baris = df_dataset.loc[[id_sampel]]

        if not baris.empty:
            nilai_input = {nama: float(baris.iloc[0][nama]) for nama in FITUR}
            diagnosis = str(baris.iloc[0]["diagnosis"]).upper()
            diagnosis_nama = "Malignant" if diagnosis == "M" else "Benign"
            st.info(
                f"**Diagnosis asli dataset: {diagnosis_nama}** — ini adalah label asli, "
                "bukan hasil prediksi model."
            )

            actual_label_sample = diagnosis_nama

            tabel_sampel = pd.DataFrame(
                {
                    "Nama fitur": [label_fitur(nama) for nama in FITUR],
                    "Nama asli": FITUR,
                    "Nilai": [nilai_input[nama] for nama in FITUR],
                }
            )
            st.dataframe(tabel_sampel, hide_index=True, use_container_width=True, height=420)

else:
    uploaded = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_csv = pd.read_csv(uploaded)
        except Exception as exc:
            raw_csv = None
            st.error("File CSV tidak dapat dibaca.")

        if raw_csv is not None:
            normalized_csv = raw_csv.copy()
            normalized_csv.columns = [
                str(column).strip().lower().replace(" ", "_")
                for column in normalized_csv.columns
            ]
            duplicate_names = sorted([
                name for name in set(normalized_csv.columns)
                if list(normalized_csv.columns).count(name) > 1
            ])
            missing = [name for name in FITUR if name not in normalized_csv.columns]
            extra = [name for name in normalized_csv.columns if name not in FITUR]

            if duplicate_names:
                st.error("Nama kolom duplikat setelah normalisasi: " + ", ".join(duplicate_names))
            elif missing or extra:
                if missing:
                    st.error("Fitur wajib hilang: " + ", ".join(missing))
                if extra:
                    st.error("Fitur tambahan tidak diperbolehkan: " + ", ".join(extra))
            else:
                csv_batch_dataframe = normalized_csv[FITUR].copy()
                st.success(f"CSV siap diproses sebagai batch: {len(csv_batch_dataframe)} baris.")
            st.dataframe(raw_csv.head(20), hide_index=True, use_container_width=True)

st.markdown('<div class="section-heading">Hasil Prediksi</div>', unsafe_allow_html=True)

respons_prediksi = None
fitur_shap: dict | None = None
if st.button(
    "Periksa dan jalankan prediksi",
    type="primary",
    use_container_width=True,
    disabled=status_model != BackendStatus.READY,
):
    if metode_input == "Upload CSV":
        if csv_batch_dataframe is None:
            st.warning("Tidak ada CSV batch yang siap diprediksi.")
        else:
            respons_prediksi = predict_batch(csv_batch_dataframe, nama_model, "csv")
            tampilkan_area_hasil(respons_prediksi)

            if respons_prediksi["status"] in ("success", "partial_success"):
                token = respons_prediksi.get("provenance_token")
                storage = jalankan_penyimpanan_sekali(
                    st.session_state,
                    f"batch:{token}",
                    lambda: simpan_batch_prediksi_terverifikasi(respons_prediksi, csv_batch_dataframe),
                )
                if storage["status"] == "saved":
                    st.success(f"{len(storage['record_ids'])} hasil batch berhasil disimpan ke riwayat.")
                elif storage["status"] == "storage_error":
                    st.warning(f"Prediksi berhasil, tetapi riwayat gagal disimpan: {storage['error']}")
    elif nilai_input is None:
        st.warning("Tidak ada data input yang siap diprediksi.")
    else:
        if set(FITUR).issubset(nilai_input.keys()):
            error_input = validasi_satu_pasien({nama: nilai_input[nama] for nama in FITUR})
            if error_input:
                st.error("\n".join(error_input))
            else:
                input_method = "sample" if metode_input == "Pilih Sampel Dataset" else "manual"
                respons_prediksi = prediksi_pasien(
                    nama_model,
                    {nama: float(nilai_input[nama]) for nama in FITUR},
                    input_method,
                )
                tampilkan_area_hasil(respons_prediksi)

                if respons_prediksi.get("status") == "success":
                    fitur_shap = {nama: float(nilai_input[nama]) for nama in FITUR}
                    token = respons_prediksi.get("provenance_token")
                    storage = jalankan_penyimpanan_sekali(
                        st.session_state,
                        f"single:{token}",
                        lambda: simpan_prediksi_terverifikasi(
                            respons_prediksi,
                            {nama: float(nilai_input[nama]) for nama in FITUR},
                            input_method,
                            actual_label=actual_label_sample,
                        ),
                    )
                    if storage["status"] == "saved":
                        st.success("Prediksi berhasil disimpan ke riwayat.")
                    elif storage["status"] == "storage_error":
                        st.warning(f"Prediksi berhasil, tetapi riwayat gagal disimpan: {storage['error']}")
        else:
            respons_prediksi = {
                "status": "error",
                "model": nama_model,
                "error": "Urutan atau nama fitur input tidak lengkap.",
                "model_status": status_model.value,
            }
            tampilkan_area_hasil(respons_prediksi)

if respons_prediksi is None:
    tampilkan_area_hasil(None)

st.markdown('<div class="section-heading">Area Interpretasi Model</div>', unsafe_allow_html=True)
col_rf, col_svm = st.columns(2)
with col_rf:
    path_fi = ROOT / "models" / "feature_importance_rf.csv"
    if path_fi.exists():
        try:
            tabel_fi = pd.read_csv(path_fi)
            st.markdown("**Feature Importance — Random Forest**")
            st.dataframe(tabel_fi.head(10), hide_index=True, use_container_width=True)
        except Exception as exc:
            tampilkan_empty_state(
                "Feature Importance tidak dapat dibaca",
                f"Artefak ditemukan tetapi gagal dibaca: {exc}",
                icon="⚠️",
            )
    else:
        tampilkan_empty_state(
            "Feature Importance belum tersedia",
            "Artefak feature_importance_rf.csv belum tersedia.",
            icon="🌲",
        )

with col_svm:
    st.markdown("**SHAP Lokal — Support Vector Machine**")
    path_bg = ROOT / "models" / "shap_background.pkl"
    if not path_bg.exists():
        tampilkan_empty_state(
            "SHAP lokal belum tersedia",
            "Artefak shap_background.pkl belum tersedia. Jalankan train.py terlebih dahulu.",
            icon="🔬",
        )
    elif nama_model != "Support Vector Machine":
        st.info("SHAP lokal dihitung untuk model SVM. Pilih model **Support Vector Machine** lalu jalankan prediksi.")
    elif metode_input == "Upload CSV":
        st.info("SHAP lokal ditampilkan untuk satu pasien (Input Manual atau Pilih Sampel Dataset).")
    elif fitur_shap is None:
        st.info("Jalankan prediksi satu pasien untuk melihat kontribusi setiap fitur terhadap hasil prediksi.")
    else:
        st.caption("Penjelasan SHAP untuk pasien yang baru diprediksi ditampilkan di bawah.")

if fitur_shap is not None and nama_model == "Support Vector Machine" and (ROOT / "models" / "shap_background.pkl").exists():
    st.markdown('<div class="section-heading">SHAP Lokal — Penjelasan Prediksi Pasien Ini</div>', unsafe_allow_html=True)
    try:
        with st.spinner("Menghitung nilai SHAP untuk pasien ini..."):
            explainer = muat_explainer_shap()
            x_scaled = preprocess_single("Support Vector Machine", fitur_shap)
            hasil_lokal = shap_svm.jelaskan_lokal(explainer, x_scaled)
        st.markdown(
            "Grafik menunjukkan fitur yang paling memengaruhi prediksi SVM untuk pasien ini. "
            "Batang **oranye** (SHAP positif) mendorong ke **Malignant**, batang **hijau** (SHAP negatif) mendorong ke **Benign**."
        )
        tampilkan_shap_lokal(hasil_lokal, [fitur_shap[nama] for nama in FITUR], top=10, kunci="prediksi")
    except Exception as exc:
        st.warning(f"Prediksi berhasil, tetapi SHAP lokal gagal dihitung: {exc}")

st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
st.markdown(
    "Sistem ini merupakan prototipe penelitian dan bukan alat diagnosis medis. "
    "Prediksi hanya dapat dijalankan ketika status backend model READY berdasarkan marker formal."
)
st.markdown('</div>', unsafe_allow_html=True)
