"""Halaman Riwayat Prediksi: menampilkan, menyaring, dan mengunduh hasil prediksi dari basis data SQLite."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from basis_data.database import ambil_detail, baca_riwayat, inisialisasi_database
from preprocessing.data import FITUR, KARAKTERISTIK
from tampilan import grafik, sumber
from tampilan.gaya import catatan, kaki, kartu_angka, kartu_hasil, kepala, tautan



def _wib(nilai) -> str:
    w = pd.to_datetime(nilai, utc=True, errors="coerce")
    return w.tz_convert("Asia/Jakarta").strftime("%d-%m-%Y %H:%M:%S WIB") if pd.notna(w) else "—"

kepala(
    "Riwayat Prediksi",
    "Semua hasil prediksi yang pernah dijalankan tersimpan otomatis di basis data SQLite lokal. "
    "Gunakan filter untuk mencari, klik pasien untuk melihat detail, dan unduh sebagai CSV.",
)

try:
    db_path = inisialisasi_database()
    semua = baca_riwayat(db_path=db_path)
except Exception as exc:  # pragma: no cover - ditampilkan ke pengguna
    st.error(f"Basis data riwayat tidak dapat dibuka: {exc}")
    st.stop()

if semua.empty:
    st.info("Belum ada riwayat prediksi. Jalankan prediksi pada menu **Prediksi**; hasilnya akan muncul di sini.")
    tautan("halaman/2_Prediksi.py", "Buka menu Prediksi", ":material/biotech:")
    kaki()
    st.stop()

waktu = pd.to_datetime(semua["Tanggal dan waktu"], utc=True, errors="coerce").dt.tz_convert("Asia/Jakarta")
semua = semua.assign(_waktu=waktu, _tanggal=waktu.dt.date)

# ---------------------------------------------------------------- Ringkasan
dengan_label = semua[semua["Label aktual dataset"].notna()]
k = st.columns(4)
with k[0]:
    kartu_angka("Total prediksi", f"{len(semua)}", f"{semua['Model'].nunique()} model tercatat")
with k[1]:
    kartu_angka("Hasil Benign", f"{(semua['Hasil klasifikasi'] == 'Benign').sum()}")
with k[2]:
    kartu_angka("Hasil Malignant", f"{(semua['Hasil klasifikasi'] == 'Malignant').sum()}", kuning=True)
with k[3]:
    if len(dengan_label):
        sesuai = (dengan_label["Hasil klasifikasi"] == dengan_label["Label aktual dataset"]).mean()
        kartu_angka("Sesuai label asli", f"{sesuai:.1%}", f"dari {len(dengan_label)} prediksi berlabel")
    else:
        kartu_angka("Sesuai label asli", "—", "hanya untuk contoh dataset")

# ---------------------------------------------------------------- Filter
st.markdown("### Filter")
f1, f2, f4 = st.columns([1, 1, 1.4])
model_f = f1.selectbox("Model", ["Semua"] + sorted(semua["Model"].unique()))
hasil_f = f2.selectbox("Hasil", ["Semua", "Benign", "Malignant"])
tgl_min, tgl_max = semua["_tanggal"].min(), semua["_tanggal"].max()
rentang_tgl = f4.date_input("Rentang tanggal", value=(tgl_min, tgl_max), min_value=tgl_min, max_value=tgl_max)
cari = st.text_input("Cari ID riwayat", placeholder="contoh: 12")

data = semua
if model_f != "Semua":
    data = data[data["Model"] == model_f]
if hasil_f != "Semua":
    data = data[data["Hasil klasifikasi"] == hasil_f]
if isinstance(rentang_tgl, (tuple, list)) and len(rentang_tgl) == 2:
    data = data[(data["_tanggal"] >= rentang_tgl[0]) & (data["_tanggal"] <= rentang_tgl[1])]
if cari.strip():
    data = data[data["ID"].astype(str).str.contains(cari.strip(), regex=False)]

st.caption(f"Menampilkan **{len(data)}** dari {len(semua)} riwayat.")
if data.empty:
    st.warning("Tidak ada riwayat yang sesuai filter. Ubah atau kosongkan filter.")
else:
    tampil = data.assign(Waktu=data["_waktu"].dt.strftime("%d-%m-%Y %H:%M WIB"))[
        ["ID", "ID pasien", "Waktu", "Model", "Hasil klasifikasi", "Probabilitas Malignant", "Label aktual dataset"]
    ].rename(columns={"Label aktual dataset": "Label asli"})
    tampil["Label asli"] = tampil["Label asli"].fillna("—")
    st.dataframe(
        tampil,
        hide_index=True,
        use_container_width=True,
        height=min(460, 38 + 35 * len(tampil)),
        column_config={
            "Probabilitas Malignant": st.column_config.ProgressColumn(
                "Probabilitas Malignant", min_value=0.0, max_value=1.0, format="%.3f"),
            "ID": st.column_config.NumberColumn("ID prediksi", format="%d"),
            "ID pasien": st.column_config.NumberColumn("ID pasien", format="%d"),
        },
    )
    st.download_button(":material/download: Unduh riwayat (CSV)", tampil.to_csv(index=False).encode("utf-8"),
                       "riwayat_prediksi.csv", "text/csv")

    # ------------------------------------------------------------ Detail
    st.markdown("### Detail riwayat")
    pilih = st.selectbox(
        "Pilih ID riwayat",
        options=data["ID"].tolist(),
        format_func=lambda i: f"ID {i} — " + " · ".join(
            str(v) for v in data.loc[data["ID"] == i, ["Model", "Hasil klasifikasi"]].iloc[0]),
    )
    detail = ambil_detail(int(pilih), db_path=db_path)
    if detail is None:
        st.error("Data riwayat tidak ditemukan.")
    else:
        kiri, kanan = st.columns([1, 1.3])
        with kiri:
            kartu_hasil(detail["Model"], detail["Hasil klasifikasi"],
                        float(detail["Probabilitas Benign"] or 0), float(detail["Probabilitas Malignant"] or 0))
            w = pd.to_datetime(detail["Tanggal dan waktu"], utc=True, errors="coerce")
            st.markdown(
                f"- **Waktu:** {w.tz_convert('Asia/Jakarta').strftime('%d-%m-%Y %H:%M:%S WIB') if pd.notna(w) else '—'}\n"
                f"- **ID pasien:** {detail['ID pasien']}\n"
                f"- **Waktu input data pasien:** {_wib(detail['Waktu input'])}\n"
                f"- **Label asli dataset:** {detail['Label aktual dataset'] or '— (bukan dari dataset)'}"
            )
        fitur = detail["Fitur"]
        with kanan:
            if all(f in fitur for f in FITUR):
                st.dataframe(pd.DataFrame({
                    "Karakteristik": [sumber.ARTI_KARAKTERISTIK[k][0] for k in KARAKTERISTIK],
                    "Mean": [fitur[f"{k}_mean"] for k in KARAKTERISTIK],
                    "SE": [fitur[f"{k}_se"] for k in KARAKTERISTIK],
                    "Worst": [fitur[f"{k}_worst"] for k in KARAKTERISTIK],
                }), hide_index=True, use_container_width=True)
        if all(f in fitur for f in FITUR) and sumber.shap_tersedia():
            if st.toggle("Tampilkan penjelasan SHAP (SVM) untuk data ini", key=f"shap_riwayat_{pilih}"):
                lokal = sumber.shap_lokal(tuple(float(fitur[f]) for f in FITUR))
                tabel = sumber.tabel_shap(lokal["shap"], [float(fitur[f]) for f in FITUR])
                catatan(sumber.kalimat_shap(tabel))
                st.altair_chart(grafik.shap_lokal(tabel, top=10, judul=f"SHAP Lokal — Riwayat ID {pilih} (SVM)"),
                          use_container_width=True)

kaki()
