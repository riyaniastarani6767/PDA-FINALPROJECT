"""Halaman Prediksi (kebutuhan fungsional F-01 s.d. F-08).

Alur sesuai DFD Level 1 laporan:
input & validasi data -> preprocessing (scaler data latih) -> klasifikasi (RF dan/atau SVM)
-> simpan ke riwayat (SQLite) -> interpretasi (SHAP lokal SVM + Feature Importance RF).
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st

from preprocessing.data import FITUR, KARAKTERISTIK, label_fitur, validasi_csv
from tampilan import grafik, sumber
from tampilan.gaya import catatan, e, kaki, kartu_angka, kartu_hasil, kepala, langkah
from utils.backend_adapter import (
    BackendStatus,
    cek_status_model,
    predict_batch,
    prediksi_pasien,
    simpan_batch_prediksi_terverifikasi,
    simpan_prediksi_terverifikasi,
)
from utils.ui_storage import jalankan_penyimpanan_sekali

PILIHAN_MODEL = {
    "Bandingkan Random Forest & SVM": ["Random Forest", "Support Vector Machine"],
    "Random Forest saja": ["Random Forest"],
    "SVM saja": ["Support Vector Machine"],
}
METODE = [":material/table_rows: Pilih contoh pasien dari dataset", ":material/edit_note: Isi manual 30 fitur",
          ":material/upload_file: Unggah file CSV (banyak pasien)"]
SINGKAT = {"Random Forest": "RF", "Support Vector Machine": "SVM"}
BATAS_BARIS_CSV = 5000

st.session_state.setdefault("hasil_prediksi", None)


def _tanda(obj) -> str:
    if isinstance(obj, pd.DataFrame):
        return hashlib.md5(pd.util.hash_pandas_object(obj, index=False).values.tobytes()).hexdigest()
    return hashlib.md5(repr(obj).encode()).hexdigest()


def _tabel_fitur_pasien(nilai: dict) -> pd.DataFrame:
    """30 fitur sebagai tabel 10 karakteristik x (Mean, SE, Worst)."""
    return pd.DataFrame(
        {
            "Karakteristik": [f"{sumber.ARTI_KARAKTERISTIK[k][0]} ({k.replace('_', ' ')})" for k in KARAKTERISTIK],
            "Mean": [nilai[f"{k}_mean"] for k in KARAKTERISTIK],
            "Standard Error": [nilai[f"{k}_se"] for k in KARAKTERISTIK],
            "Worst": [nilai[f"{k}_worst"] for k in KARAKTERISTIK],
        }
    )


# ---------------------------------------------------------------------------
kepala(
    "Prediksi Tumor Payudara",
    "Ikuti 3 langkah di bawah: pilih model, masukkan data pasien, lalu jalankan prediksi. "
    "Hasil dan penjelasannya muncul di bagian bawah dan otomatis tersimpan ke Riwayat Prediksi.",
)

status = {m: cek_status_model(m) for m in sumber.MODEL_UI}
model_siap = [m for m, s in status.items() if s == BackendStatus.READY]
if not model_siap:
    st.error(
        "Model belum siap digunakan. Jalankan `python train.py` lalu `python verify_models.py` "
        "di folder proyek, kemudian muat ulang halaman ini."
    )
    st.stop()

# ---------------------------------------------------------------- Langkah 1
langkah(1, "Pilih model klasifikasi", "Disarankan membandingkan kedua model agar hasilnya bisa dicek silang.")
pilihan = st.radio(
    "Model yang digunakan",
    list(PILIHAN_MODEL),
    horizontal=True,
    key="pilihan_model",
    captions=[
        "Prediksi dengan dua model sekaligus",
        "Ensemble 100 pohon keputusan",
        "Kernel RBF — model terbaik (recall tertinggi)",
    ],
    label_visibility="collapsed",
)
model_dipakai = [m for m in PILIHAN_MODEL[pilihan] if m in model_siap]
tidak_siap = [m for m in PILIHAN_MODEL[pilihan] if m not in model_siap]
if tidak_siap:
    st.warning("Model berikut belum siap dan dilewati: " + ", ".join(tidak_siap))

# ---------------------------------------------------------------- Langkah 2
langkah(2, "Masukkan data pasien", "Pilih salah satu cara di bawah ini.")
metode = st.radio("Cara memasukkan data", METODE, horizontal=True, key="metode_input", label_visibility="collapsed")

df_data = sumber.dataset()
rentang = sumber.rentang_fitur()
input_tunggal: dict | None = None
label_asli: str | None = None
metode_db = "sample"
batch_df: pd.DataFrame | None = None
batch_info: dict = {}

if metode == METODE[0]:
    st.info("Gunakan data pasien asli dari dataset untuk mencoba sistem. Label diagnosis aslinya ditampilkan "
            "agar hasil prediksi dapat dibandingkan.")
    c1, c2 = st.columns([1, 2])
    with c1:
        saring = st.radio("Tampilkan pasien", ["Semua", "Benign", "Malignant"], horizontal=True, key="saring_sampel")
    pilihan_df = df_data if saring == "Semua" else df_data[df_data.label == saring]
    with c2:
        idx = st.selectbox(
            "Pilih pasien",
            options=pilihan_df.index.tolist(),
            format_func=lambda i: f"Pasien ID {df_data.at[i, 'id']} — diagnosis asli: {df_data.at[i, 'label']}",
            key="pilih_sampel",
        )
    baris = df_data.loc[idx]
    input_tunggal = {f: float(baris[f]) for f in FITUR}
    label_asli = str(baris["label"])
    with st.expander(f"Lihat 30 nilai fitur pasien ID {baris['id']}", expanded=False):
        st.dataframe(_tabel_fitur_pasien(input_tunggal), hide_index=True, use_container_width=True)

elif metode == METODE[1]:
    metode_db = "manual"

    def _isi(sumber_nilai: dict | None) -> None:
        for f in FITUR:
            st.session_state[f"m_{f}"] = None if sumber_nilai is None else float(sumber_nilai[f])

    contoh_b = df_data[df_data.diagnosis == "B"].iloc[0]
    contoh_m = df_data[df_data.diagnosis == "M"].iloc[0]
    st.markdown("Isi ke-30 nilai fitur hasil analisis citra FNA. Arahkan kursor ke ikon **?** untuk melihat arti "
                "fitur dan rentang nilainya pada dataset. Tombol di bawah dapat dipakai untuk mengisi contoh.")
    t1, t2, t3, t4 = st.columns(4)
    t1.button("Isi contoh Benign", on_click=_isi, args=({f: contoh_b[f] for f in FITUR},), use_container_width=True)
    t2.button("Isi contoh Malignant", on_click=_isi, args=({f: contoh_m[f] for f in FITUR},), use_container_width=True)
    t3.button("Isi nilai median dataset", on_click=_isi, args=(rentang["median"].to_dict(),), use_container_width=True)
    t4.button("Kosongkan semua", on_click=_isi, args=(None,), use_container_width=True)

    nilai_manual = {}
    tab_mean, tab_se, tab_worst = st.tabs(["Mean (rata-rata)", "Standard Error (SE)", "Worst (terburuk)"])
    for tab, variasi in ((tab_mean, "mean"), (tab_se, "se"), (tab_worst, "worst")):
        with tab:
            kolom = st.columns(2)
            for i, k in enumerate(KARAKTERISTIK):
                f = f"{k}_{variasi}"
                r = rentang.loc[f]
                with kolom[i % 2]:
                    nilai_manual[f] = st.number_input(
                        f"{sumber.ARTI_KARAKTERISTIK[k][0]} — {label_fitur(f)}",
                        min_value=0.0,
                        value=None,
                        step=float(10 ** np.floor(np.log10(max(r["median"], 1e-6))) / 10),
                        format="%.5f",
                        key=f"m_{f}",
                        placeholder=f"contoh: {r['median']:.4g}",
                        help=f"{sumber.arti_fitur(f)} Rentang pada dataset: {r['min']:.4g} – {r['max']:.4g}.",
                    )
    terisi = sum(v is not None for v in nilai_manual.values())
    if terisi < len(FITUR):
        kosong = [label_fitur(f) for f, v in nilai_manual.items() if v is None]
        st.warning(f"Baru {terisi} dari 30 fitur terisi. Belum diisi: {', '.join(kosong[:6])}"
                   f"{' …' if len(kosong) > 6 else ''}. Periksa ketiga tab (Mean, SE, Worst).")
    else:
        st.success("Ke-30 fitur sudah terisi. Lanjutkan ke langkah 3.")
        input_tunggal = {f: float(v) for f, v in nilai_manual.items()}
        luar = [label_fitur(f) for f, v in input_tunggal.items()
                if v < rentang.at[f, "min"] * 0.5 or v > rentang.at[f, "max"] * 1.5]
        if luar:
            catatan("Beberapa nilai jauh di luar rentang dataset (" + ", ".join(luar[:5]) +
                    "). Prediksi tetap dapat dijalankan, tetapi hasilnya kurang dapat dipercaya.", kuning=True)

else:
    metode_db = "csv"
    st.markdown("Unggah file **CSV** berisi satu baris per pasien dengan **30 kolom fitur** (nama kolom sama seperti "
                "dataset). Kolom `id` dan `diagnosis` boleh ada; kolom `diagnosis` (M/B) dipakai untuk membandingkan "
                "hasil prediksi dengan label asli.")
    d1, d2 = st.columns(2)
    d1.download_button(":material/download: Unduh template CSV (30 kolom, 2 contoh pasien)", sumber.template_csv(),
                       "template_input_prediksi.csv", "text/csv", use_container_width=True)
    d2.download_button(":material/download: Unduh contoh CSV 20 pasien (dengan id & diagnosis)", sumber.contoh_csv_dataset(20),
                       "contoh_20_pasien.csv", "text/csv", use_container_width=True)
    berkas = st.file_uploader("Unggah file CSV", type=["csv"], key="unggah_csv")
    if berkas is None:
        st.info("Belum ada file yang diunggah. Anda dapat mencoba dengan file contoh di atas.")
    else:
        mentah = None
        try:
            isi = berkas.getvalue()
            if not isi.strip():
                raise ValueError("File kosong.")
            mentah = pd.read_csv(io.BytesIO(isi), sep=None, engine="python", encoding="utf-8-sig")
        except Exception as exc:
            st.error(f"File tidak dapat dibaca sebagai CSV ({exc}). Pastikan file berformat .csv dengan pemisah koma "
                     "atau titik koma, lalu coba lagi.")
        if mentah is not None:
            data_valid, kesalahan = validasi_csv(mentah)
            kolom_norm = [str(c).strip().lower().replace(" ", "_") for c in mentah.columns]
            diabaikan = [c for c in kolom_norm if c not in FITUR and not c.startswith("unnamed")]
            if kesalahan:
                for pesan in kesalahan:
                    st.error(pesan)
                st.caption("Perbaiki file lalu unggah ulang. Gunakan template CSV di atas sebagai acuan nama kolom.")
            elif len(data_valid) > BATAS_BARIS_CSV:
                st.error(f"File berisi {len(data_valid)} baris. Maksimal {BATAS_BARIS_CSV} pasien per unggahan.")
            else:
                batch_df = data_valid
                norm = mentah.copy()
                norm.columns = kolom_norm
                if "id" in norm.columns:
                    batch_info["id"] = norm["id"].astype(str).tolist()
                if "diagnosis" in norm.columns:
                    batch_info["label_asli"] = (
                        norm["diagnosis"].astype(str).str.strip().str.upper()
                        .map({"M": "Malignant", "B": "Benign", "MALIGNANT": "Malignant", "BENIGN": "Benign"})
                        .astype(object).where(lambda s: s.notna(), None).tolist()
                    )
                st.success(f"File valid: **{len(batch_df)} pasien** siap diprediksi.")
                if diabaikan:
                    st.info("Kolom berikut tidak dipakai sebagai fitur model: " + ", ".join(f"`{c}`" for c in diabaikan)
                            + (". Kolom `diagnosis` hanya dipakai sebagai pembanding." if "diagnosis" in diabaikan else "."))
                with st.expander("Pratinjau data (10 baris pertama)"):
                    st.dataframe(mentah.head(10), use_container_width=True)

# ---------------------------------------------------------------- Langkah 3
langkah(3, "Jalankan prediksi")
ada_input = input_tunggal is not None or batch_df is not None
kunci_input = _tanda((metode, tuple(model_dipakai),
                      tuple(input_tunggal.values()) if input_tunggal else None,
                      _tanda(batch_df) if batch_df is not None else None))
jalan = st.button("Jalankan prediksi", icon=":material/play_arrow:", type="primary", use_container_width=True,
                  disabled=not ada_input or not model_dipakai)
if not ada_input:
    st.caption("Tombol aktif setelah data pasien lengkap/valid pada langkah 2.")

if jalan:
    with st.spinner("Memproses data dan menjalankan model..."):
        if batch_df is not None:
            respons = {m: predict_batch(batch_df, m, "csv") for m in model_dipakai}
            tersimpan = {}
            for m, r in respons.items():
                if r["status"] in ("success", "partial_success"):
                    tersimpan[m] = jalankan_penyimpanan_sekali(
                        st.session_state, f"batch:{r.get('provenance_token')}",
                        lambda r=r: simpan_batch_prediksi_terverifikasi(r, batch_df))
            st.session_state.hasil_prediksi = {"jenis": "batch", "kunci": kunci_input, "respons": respons,
                                               "data": batch_df, "info": batch_info, "simpan": tersimpan}
        else:
            respons = {m: prediksi_pasien(m, input_tunggal, metode_db) for m in model_dipakai}
            tersimpan = {}
            for m, r in respons.items():
                if r.get("status") == "success":
                    tersimpan[m] = jalankan_penyimpanan_sekali(
                        st.session_state, f"single:{r.get('provenance_token')}",
                        lambda r=r: simpan_prediksi_terverifikasi(r, input_tunggal, metode_db, actual_label=label_asli))
            st.session_state.hasil_prediksi = {"jenis": "tunggal", "kunci": kunci_input, "respons": respons,
                                               "fitur": input_tunggal, "label_asli": label_asli, "simpan": tersimpan}

hasil = st.session_state.hasil_prediksi
if hasil is not None and hasil["kunci"] != kunci_input:
    if ada_input:
        st.info("Data atau model berubah sejak prediksi terakhir. Klik **Jalankan prediksi** untuk memperbarui hasil.")
    hasil = None


def _pesan_simpan(simpan: dict) -> None:
    ok = [m for m, s in simpan.items() if s.get("status") in ("saved", "already_saved")]
    gagal = {m: s.get("error") for m, s in simpan.items() if s.get("status") == "storage_error"}
    if ok:
        st.caption(":material/save: Hasil tersimpan ke **Riwayat Prediksi** (" + ", ".join(SINGKAT[m] for m in ok) + ").")
    for m, err in gagal.items():
        st.warning(f"Prediksi {m} berhasil, tetapi gagal disimpan ke riwayat: {err}")


def _tampilkan_shap(fitur: dict, judul_pasien: str, kunci: str) -> None:
    if not sumber.shap_tersedia():
        st.warning("Artefak SHAP belum tersedia (models/shap_background.pkl). Jalankan train.py terlebih dahulu.")
        return
    with st.spinner("Menghitung nilai SHAP..."):
        lokal = sumber.shap_lokal(tuple(fitur[f] for f in FITUR))
    tabel = sumber.tabel_shap(lokal["shap"], [fitur[f] for f in FITUR])
    catatan(sumber.kalimat_shap(tabel))
    st.altair_chart(grafik.shap_lokal(tabel, top=10, judul=f"10 Fitur Paling Berpengaruh — {judul_pasien} (SVM)"),
                    use_container_width=True)
    st.caption(
        f"Cara membaca: batang **kuning** mendorong prediksi ke **Malignant**, batang **hijau** ke **Benign**; "
        f"makin panjang batang, makin besar pengaruhnya. Nilai dasar (rata-rata) {lokal['base_value']:.3f} + jumlah "
        f"seluruh nilai SHAP = {lokal['base_value'] + sum(lokal['shap']):.3f}, sama dengan probabilitas Malignant "
        f"dari SVM = {lokal['prediksi']:.3f}."
    )
    with st.expander("Lihat nilai SHAP ke-30 fitur"):
        st.dataframe(tabel.style.format({"Nilai fitur": "{:.5g}", "Nilai SHAP": "{:+.5f}"}),
                     hide_index=True, use_container_width=True, key=f"tabel_shap_{kunci}")


def _tampilkan_fi() -> None:
    fi = sumber.feature_importance()
    if fi is None:
        st.warning("Feature importance belum tersedia (models/feature_importance_rf.csv).")
        return
    top = fi.head(10)
    fig = grafik.batang_horizontal(top["nama"], top["importance"], "10 Fitur Terpenting — Random Forest",
                                   "Feature importance (mean decrease impurity)")
    st.altair_chart(fig, use_container_width=True)
    st.caption("Feature importance bersifat **global**: menunjukkan fitur yang paling sering dan paling efektif "
               "dipakai pohon-pohon Random Forest untuk memisahkan kelas pada seluruh data latih.")


# ---------------------------------------------------------------- Hasil
if hasil is not None and hasil["jenis"] == "tunggal":
    st.markdown("## Hasil prediksi")
    respons = hasil["respons"]
    kolom = st.columns(len(respons))
    label_ok = []
    for kol, (m, r) in zip(kolom, respons.items()):
        with kol:
            if r.get("status") == "success":
                kartu_hasil(m, r["classification_label"], r["probability_benign"], r["probability_malignant"])
                label_ok.append(r["classification_label"])
            else:
                st.error(f"{m}: prediksi gagal — {r.get('error', 'kesalahan tidak diketahui')}")
    if len(label_ok) == 2:
        if label_ok[0] == label_ok[1]:
            catatan(f"Kedua model <b>sepakat</b>: tumor diprediksi <b>{label_ok[0]}</b>.")
        else:
            catatan("Kedua model <b>berbeda pendapat</b>. Hasil perlu ditinjau lebih hati-hati; SVM memiliki recall "
                    "lebih tinggi pada evaluasi (lebih jarang melewatkan kasus Malignant).", kuning=True)
    if hasil["label_asli"] and label_ok:
        cocok = all(l == hasil["label_asli"] for l in label_ok)
        st.markdown(f"Diagnosis asli pasien pada dataset: **{hasil['label_asli']}** — "
                    + (":material/check_circle: prediksi **sesuai**" if cocok else ":material/cancel: ada prediksi yang **tidak sesuai**"))
    _pesan_simpan(hasil["simpan"])

    st.markdown("## Mengapa model memprediksi demikian?")
    if "Random Forest" in respons:
        tab_shap, tab_fi = st.tabs(["SHAP lokal — pasien ini (SVM)", "Feature Importance — Random Forest"])
    else:
        tab_shap, tab_fi = st.tabs(["SHAP lokal — pasien ini (SVM)", "Feature Importance — Random Forest (pembanding)"])
    with tab_shap:
        if "Support Vector Machine" not in respons:
            st.caption("Sesuai rancangan penelitian, SHAP dihitung pada model SVM untuk pasien yang sama.")
        _tampilkan_shap(hasil["fitur"], "pasien ini", "tunggal")
    with tab_fi:
        _tampilkan_fi()

elif hasil is not None and hasil["jenis"] == "batch":
    st.markdown("## Hasil prediksi CSV")
    respons, info, data = hasil["respons"], hasil["info"], hasil["data"]
    n = len(data)
    tabel = pd.DataFrame({"No": range(1, n + 1)})
    if "id" in info:
        tabel["ID"] = info["id"]
    if "label_asli" in info:
        tabel["Label asli"] = info["label_asli"]
    ringkas = st.columns(len(respons) + 1)
    with ringkas[0]:
        kartu_angka("Jumlah pasien", f"{n}")
    for i, (m, r) in enumerate(respons.items(), start=1):
        det = r.get("detail", [])
        hasil_m = [d.get("classification_label") for d in det]
        tabel[f"Hasil {SINGKAT[m]}"] = hasil_m
        tabel[f"P(Malignant) {SINGKAT[m]}"] = [d.get("probability_malignant") for d in det]
        n_m = sum(h == "Malignant" for h in hasil_m)
        n_b = sum(h == "Benign" for h in hasil_m)
        ket = f"{n_b} Benign · {n_m} Malignant"
        if "label_asli" in info:
            pas = [(h, a) for h, a in zip(hasil_m, info["label_asli"]) if h and a]
            if pas:
                ket += f" · sesuai label asli {sum(h == a for h, a in pas)}/{len(pas)}"
        with ringkas[i]:
            kartu_angka(m, f"{r.get('jumlah_berhasil', 0)} berhasil", ket, kuning=n_m > 0)
        if r.get("jumlah_gagal"):
            st.warning(f"{m}: {r['jumlah_gagal']} baris gagal diproses.")
            tabel[f"Catatan {SINGKAT[m]}"] = [d.get("error") or "" for d in det]
    if "label_asli" in info and any(info["label_asli"]):
        catatan("Kolom <b>Label asli</b> berasal dari kolom <code>diagnosis</code> pada file. Jika data berasal dari "
                "dataset penelitian, sebagian besar baris adalah data latih model, sehingga kesesuaian di sini "
                "<b>bukan</b> ukuran kinerja model (lihat menu Evaluasi Model).")

    konfig = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f")
              for c in tabel.columns if c.startswith("P(Malignant)")}
    st.dataframe(tabel, hide_index=True, use_container_width=True, column_config=konfig, height=min(420, 38 + 35 * n))
    unduh = tabel.copy()
    st.download_button(":material/download: Unduh hasil prediksi (CSV)", unduh.to_csv(index=False).encode("utf-8"),
                       "hasil_prediksi.csv", "text/csv")
    _pesan_simpan(hasil["simpan"])

    st.markdown("## Penjelasan SHAP per pasien")
    no = st.selectbox(
        "Pilih pasien untuk dijelaskan",
        options=list(range(n)),
        format_func=lambda i: f"Baris {i + 1}" + (f" — ID {info['id'][i]}" if "id" in info else "")
        + " — " + " / ".join(f"{SINGKAT[m]}: {r['detail'][i].get('classification_label') or 'gagal'}"
                             for m, r in respons.items()),
        key="pilih_baris_shap",
    )
    fitur_baris = {f: float(data.iloc[no][f]) for f in FITUR}
    _tampilkan_shap(fitur_baris, f"baris {no + 1}", f"batch_{no}")

kaki()
