from pathlib import Path

import streamlit as st
import pandas as pd

from basis_data.database import (
    ambil_detail,
    baca_riwayat,
    inisialisasi_database,
)
from components.empty_state import tampilkan_empty_state
from preprocessing.data import FITUR, label_fitur


ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "assets" / "styles.css"
st.set_page_config(page_title="Riwayat Prediksi", page_icon="🗂️", layout="wide")
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def format_tanggal_riwayat(value):
    if pd.isna(value):
        return "—"
    try:
        timestamp = pd.to_datetime(value, utc=True)
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


try:
    db_path = inisialisasi_database()
except Exception:
    db_path = None
    st.error("Database riwayat tidak dapat diinisialisasi saat ini.")

st.title("Riwayat Prediksi")

if db_path is None:
    tampilkan_empty_state(
        "Riwayat tidak dapat dibuka",
        "Database tidak dapat diakses pada saat ini.",
        icon="🗄️",
    )
else:
    try:
        semua = baca_riwayat(db_path=db_path)
    except Exception:
        semua = None
        st.error("Riwayat gagal dibaca dari database.")

    if semua is None:
        tampilkan_empty_state("Riwayat tidak tersedia", "Gagal membaca database riwayat.", icon="🗄️")
    elif semua.empty:
        st.markdown('<div class="section-heading">Ringkasan</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total prediksi", 0)
        col2.metric("Benign", 0)
        col3.metric("Malignant", 0)
        col4.metric("Model tercatat", 0)

        tampilkan_empty_state(
            "Belum ada riwayat prediksi",
            "Riwayat akan terisi setelah backend model terverifikasi dan satu prediksi benar-benar berhasil dijalankan.",
            icon="🗂️",
        )
    else:
        total = len(semua)
        benign = int((semua["Hasil klasifikasi"] == "Benign").sum())
        malignant = int((semua["Hasil klasifikasi"] == "Malignant").sum())
        jumlah_model = int(semua["Model"].nunique())

        st.markdown('<div class="section-heading">Ringkasan</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total prediksi", total)
        col2.metric("Benign", benign)
        col3.metric("Malignant", malignant)
        col4.metric("Model tercatat", jumlah_model)

        st.markdown('<div class="section-heading">Filter Riwayat</div>', unsafe_allow_html=True)
        col_search, col_model, col_hasil = st.columns([2, 2, 2])
        with col_search:
            id_search = st.text_input("Cari berdasarkan ID")
        with col_model:
            model_options = ["Semua"] + sorted(semua["Model"].dropna().unique().tolist())
            model_filter = st.selectbox("Model", options=model_options)
        with col_hasil:
            hasil_options = ["Semua"] + sorted(semua["Hasil klasifikasi"].dropna().unique().tolist())
            hasil_filter = st.selectbox("Hasil", options=hasil_options)

        col_awal, col_akhir = st.columns(2)
        with col_awal:
            tanggal_awal = st.date_input("Tanggal awal", value=None)
        with col_akhir:
            tanggal_akhir = st.date_input("Tanggal akhir", value=None)

        filters = {
            "id_search": id_search.strip(),
            "model": model_filter,
            "classification": hasil_filter,
            "tanggal_awal": tanggal_awal.isoformat() if tanggal_awal else None,
            "tanggal_akhir": tanggal_akhir.isoformat() if tanggal_akhir else None,
        }

        riwayat_filter = baca_riwayat(filters=filters, db_path=db_path)

        if riwayat_filter.empty:
            tampilkan_empty_state(
                "Tidak ada riwayat sesuai filter",
                "Ubah atau hapus filter untuk menampilkan riwayat yang tersedia.",
                icon="🔎",
            )
        else:
            riwayat_tampil = riwayat_filter.copy()
            if "Tanggal dan waktu" in riwayat_tampil.columns:
                riwayat_tampil["Tanggal dan waktu"] = riwayat_tampil["Tanggal dan waktu"].apply(format_tanggal_riwayat)

            st.markdown('<div class="section-heading">Tabel Riwayat</div>', unsafe_allow_html=True)
            st.dataframe(
                riwayat_tampil.style.format(
                    {
                        "Probabilitas Benign": lambda value: "—" if pd.isna(value) else f"{value:.2%}",
                        "Probabilitas Malignant": lambda value: "—" if pd.isna(value) else f"{value:.2%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

            st.download_button(
                "Ekspor CSV sesuai filter",
                data=riwayat_tampil.to_csv(index=False).encode("utf-8-sig"),
                file_name="riwayat_prediksi.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown('<div class="section-heading">Detail Riwayat</div>', unsafe_allow_html=True)
            id_options = riwayat_filter["ID"].tolist()
            id_pilih = st.selectbox("Pilih ID", options=id_options)
            detail = ambil_detail(id_pilih, db_path=db_path)

            if detail is None:
                tampilkan_empty_state("Detail tidak ditemukan", "Record pilihan tidak tersedia.", icon="📄")
            else:
                info1, info2 = st.columns(2)
                with info1:
                    st.markdown("**Informasi prediksi**")
                    st.write(f"ID: {detail['ID']}")
                    st.write(f"Waktu: {format_tanggal_riwayat(detail['Tanggal dan waktu'])}")
                    st.write(f"Model: {detail['Model']}")
                    st.write(f"Metode input: {detail['Metode input']}")
                with info2:
                    st.markdown("**Hasil**")
                    st.write(f"Hasil klasifikasi: {detail['Hasil klasifikasi']}")
                    if detail["Probabilitas Benign"] is None:
                        st.write("Probabilitas Benign: tidak tersedia")
                    else:
                        st.write(f"Probabilitas Benign: {detail['Probabilitas Benign']:.2%}")
                    if detail["Probabilitas Malignant"] is None:
                        st.write("Probabilitas Malignant: tidak tersedia")
                    else:
                        st.write(f"Probabilitas Malignant: {detail['Probabilitas Malignant']:.2%}")
                    if detail["Label aktual dataset"]:
                        st.caption(
                            f"Label aktual dataset: {detail['Label aktual dataset']} "
                            "— ini bukan hasil prediksi model."
                        )

                st.markdown("**30 fitur input**")
                fitur = detail["Fitur"]
                tabel_fitur = []
                for nama in FITUR:
                    tabel_fitur.append({
                        "Nama asli": nama,
                        "Nama tampilan": label_fitur(nama),
                        "Nilai": fitur.get(nama),
                    })

                st.dataframe(
                    pd.DataFrame(tabel_fitur),
                    hide_index=True,
                    use_container_width=True,
                    height=420,
                )

st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
st.markdown(
    "Riwayat hanya berisi hasil prediksi yang benar-benar berhasil dijalankan oleh backend terverifikasi. "
    "Tidak ada identitas atau informasi pribadi pasien yang disimpan."
)
st.markdown('</div>', unsafe_allow_html=True)
