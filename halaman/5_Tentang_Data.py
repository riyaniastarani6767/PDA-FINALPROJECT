"""Halaman Tentang Data: informasi dataset Breast Cancer Wisconsin (Diagnostic) dan tahapan preprocessing."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from preprocessing.data import FITUR, label_fitur
from tampilan import grafik, sumber
from tampilan.gaya import catatan, kaki, kartu_angka, kepala, langkah

kepala(
    "Tentang Data",
    "Dataset <b>Breast Cancer Wisconsin (Diagnostic)</b> dari UCI Machine Learning Repository: hasil pengukuran "
    "inti sel dari citra digital sampel <i>Fine Needle Aspiration</i> (FNA) massa payudara.",
)

df = sumber.dataset()
eda = sumber.hasil_eda() or {}
kual = eda.get("kualitas", {})
n_b, n_m = int((df.diagnosis == "B").sum()), int((df.diagnosis == "M").sum())

k = st.columns(4)
with k[0]:
    kartu_angka("Jumlah pasien", f"{len(df)}", "1 baris = 1 pasien")
with k[1]:
    kartu_angka("Jumlah fitur", "30", "10 karakteristik × 3 variasi")
with k[2]:
    kartu_angka("Benign (jinak)", f"{n_b}", f"{n_b / len(df):.1%}")
with k[3]:
    kartu_angka("Malignant (ganas)", f"{n_m}", f"{n_m / len(df):.1%}", kuning=True)

t1, t2, t3, t4 = st.tabs([":material/list_alt: Fitur dataset", ":material/query_stats: Eksplorasi data",
                      ":material/tune: Preprocessing", ":material/table_view: Lihat dataset"])

with t1:
    st.markdown("Setiap pasien memiliki **10 karakteristik inti sel**. Masing-masing dihitung dalam tiga variasi:")
    v1, v2, v3 = st.columns(3)
    v1.markdown("**Mean** — " + sumber.ARTI_VARIASI["mean"])
    v2.markdown("**SE** — " + sumber.ARTI_VARIASI["se"])
    v3.markdown("**Worst** — " + sumber.ARTI_VARIASI["worst"])
    st.dataframe(sumber.daftar_karakteristik(), hide_index=True, use_container_width=True)
    catatan("Target klasifikasi adalah kolom <b>diagnosis</b>: <b>M</b> = Malignant (ganas, kelas positif = 1) dan "
            "<b>B</b> = Benign (jinak = 0). Kolom <b>id</b> hanya nomor identitas pasien dan tidak dipakai model.")
    st.markdown("Sumber: [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)]"
                "(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (Wolberg dkk., 1993).")

with t2:
    st.markdown(
        "Eksplorasi data (EDA) dilakukan sebelum pemodelan untuk menjawab lima pertanyaan: apakah data bersih, "
        "apakah kelas seimbang, bagaimana skala tiap fitur, fitur mana yang membedakan Benign dan Malignant, "
        "serta bagaimana hubungan antarfitur. Setiap temuan di bawah ini menjadi dasar langkah **Preprocessing**."
    )

    # ------------------------------------------------------------ 1. Kualitas data
    langkah(1, "Kualitas data", "Apakah ada data kosong, ganda, atau nilai yang tidak wajar?")
    n_missing = kual.get("missing_value", int(df[FITUR].isna().sum().sum()))
    st.dataframe(pd.DataFrame({
        "Pemeriksaan": ["Jumlah baris (pasien)", "Jumlah kolom", "Fitur numerik", "Missing value", "Baris duplikat",
                        "ID pasien duplikat", "Nilai negatif", "Label unik"],
        "Hasil": [str(kual.get("jumlah_data", len(df))), str(kual.get("jumlah_kolom", df.shape[1] - 1)),
                  str(kual.get("fitur_numerik", 30)), str(n_missing), str(kual.get("duplikat", 0)),
                  str(kual.get("id_duplikat", 0)), str(kual.get("nilai_negatif", 0)),
                  ", ".join(kual.get("label_unik", ["B", "M"]))],
        "Artinya": [
            "Setiap baris mewakili satu pasien (satu citra FNA).",
            "Terdiri dari id, diagnosis, dan 30 fitur pengukuran inti sel.",
            "Seluruh fitur bertipe angka sehingga dapat langsung diolah model tanpa encoding.",
            "Tidak ada nilai kosong, sehingga tidak diperlukan imputasi.",
            "Tidak ada pasien yang tercatat dua kali, sehingga tidak ada kebocoran data latih ke data uji.",
            "Setiap id unik; kolom id hanya identitas dan dihapus sebelum pelatihan.",
            "Ukuran fisik (jari-jari, luas, dll.) tidak mungkin negatif; tidak ada nilai yang tidak wajar.",
            "Hanya dua kelas: B = Benign (jinak) dan M = Malignant (ganas).",
        ],
    }), hide_index=True, use_container_width=True)
    catatan("<b>Kesimpulan:</b> dataset sudah bersih. Langkah preprocessing tidak perlu menghapus atau mengisi data, "
            "cukup membuang kolom <b>id</b> dan mengubah label menjadi angka (Malignant = 1, Benign = 0).")

    # ------------------------------------------------------------ 2. Distribusi kelas
    langkah(2, "Distribusi kelas", "Apakah jumlah pasien Benign dan Malignant seimbang?")
    dk = eda.get("distribusi_kelas", {})
    rasio = dk.get("rasio_mayoritas_minoritas", round(n_b / n_m, 2))
    a, b = st.columns([1, 1.1])
    with a:
        st.altair_chart(grafik.distribusi_kelas(n_b, n_m), use_container_width=True)
    with b:
        st.markdown(
            f"Dari **{len(df)} pasien**, sebanyak **{n_b} ({n_b / len(df):.1%})** berdiagnosis Benign dan "
            f"**{n_m} ({n_m / len(df):.1%})** Malignant, dengan rasio **{rasio} : 1**.\n\n"
            "Ketidakseimbangan ini tergolong **ringan**, tetapi tetap berpengaruh:\n"
            "- Model yang selalu menebak *Benign* sudah mendapat akurasi sekitar 63%, sehingga **akurasi saja tidak "
            "cukup** untuk menilai model. Karena itu evaluasi juga memakai precision, recall, dan F1-score.\n"
            "- Pembagian data latih dan data uji dilakukan secara **stratified** (80 : 20) agar proporsi Benign dan "
            "Malignant sama di kedua bagian.\n"
            "- **Recall** kelas Malignant diutamakan, karena salah menganggap tumor ganas sebagai jinak (false negative) "
            "lebih berbahaya daripada sebaliknya."
        )

    # ------------------------------------------------------------ 3. Skala fitur
    langkah(3, "Skala dan statistik deskriptif", "Apakah rentang nilai setiap fitur sebanding?")
    stat = sumber.eda_statistik()
    sk = eda.get("skala", {})
    a, b = st.columns([1.3, 1])
    with a:
        st.altair_chart(grafik.skala_fitur(stat.rename(index=label_fitur), "Nilai Maksimum Setiap Fitur"),
                        use_container_width=True)
    with b:
        besar, kecil = sk.get("fitur_nilai_terbesar", stat["max"].idxmax()), sk.get("fitur_nilai_terkecil", stat["max"].idxmin())
        st.markdown(
            f"Rentang nilai antarfitur sangat berbeda. Fitur **{label_fitur(besar)}** bisa mencapai "
            f"**{stat.loc[besar, 'max']:,.1f}**, sedangkan **{label_fitur(kecil)}** maksimal hanya "
            f"**{stat.loc[kecil, 'max']:.4f}**. Rentang terlebar sekitar **{sk.get('rasio_rentang', 0):,.0f} kali** "
            "lebih besar daripada rentang tersempit.\n\n"
            "Grafik di samping memakai **skala logaritmik**, sehingga setiap garis vertikal berarti nilai 10 kali lipat.\n\n"
            "**Dampaknya:** SVM dengan kernel RBF menghitung jarak antarpasien. Tanpa penyetaraan skala, fitur bernilai "
            "besar seperti *area* akan mendominasi jarak, sementara fitur kecil seperti *smoothness* hampir tidak "
            "berpengaruh. Karena itu seluruh fitur distandardisasi dengan **z-score** (rata-rata 0, simpangan baku 1) "
            "memakai μ dan σ dari data latih."
        )
        catatan("Random Forest sebenarnya tidak terpengaruh skala, karena hanya membandingkan nilai dengan ambang "
                "(threshold). Standardisasi terutama dibutuhkan oleh SVM.")
    with st.expander("Lihat statistik deskriptif 30 fitur"):
        tampil = stat.rename(index=label_fitur).rename(columns={
            "min": "Min", "max": "Maks", "mean": "Rata-rata", "median": "Median", "std": "Simpangan baku",
            "rentang": "Rentang"})
        st.dataframe(tampil.round(4), use_container_width=True, height=420)

    # ------------------------------------------------------------ 4. Outlier
    langkah(4, "Outlier (metode IQR)", "Apakah ada nilai ekstrem, dan dari kelas mana asalnya?")
    ol = eda.get("outlier", {})
    k = st.columns(4)
    with k[0]:
        kartu_angka("Nilai outlier", f"{ol.get('total_nilai_outlier', 0)}",
                    f"dari {ol.get('total_nilai', 0):,} nilai ({ol.get('total_nilai_outlier', 0) / max(ol.get('total_nilai', 1), 1):.1%})")
    with k[1]:
        kartu_angka("Pasien dengan outlier", f"{ol.get('pasien_dengan_outlier', 0)}",
                    f"{ol.get('pasien_dengan_outlier', 0) / len(df):.1%} dari seluruh pasien")
    with k[2]:
        kartu_angka("Malignant punya outlier", f"{ol.get('persen_malignant_punya_outlier', 0)}%",
                    f"{ol.get('pasien_outlier_malignant', 0)} dari {n_m} pasien", kuning=True)
    with k[3]:
        kartu_angka("Benign punya outlier", f"{ol.get('persen_benign_punya_outlier', 0)}%",
                    f"{ol.get('pasien_outlier_benign', 0)} dari {n_b} pasien")
    a, b = st.columns([1.3, 1])
    with a:
        st.altair_chart(grafik.outlier_per_fitur(sumber.eda_outlier().rename(index=label_fitur),
                                                 "12 Fitur dengan Outlier Terbanyak"), use_container_width=True)
    with b:
        st.markdown(
            "Sebuah nilai disebut outlier jika berada di luar rentang **Q1 − 1,5 × IQR** sampai **Q3 + 1,5 × IQR** "
            "(IQR = Q3 − Q1).\n\n"
            f"Hasilnya, **{ol.get('persen_nilai_outlier_dari_malignant', 0)}%** nilai outlier berasal dari pasien "
            f"**Malignant**, dan hampir semuanya (**{ol.get('di_atas_batas_atas', 0)}** dari "
            f"{ol.get('total_nilai_outlier', 0)}) berada **di atas** batas atas. Artinya, nilai ekstrem ini bukan "
            "kesalahan pencatatan. Nilai tersebut menggambarkan ciri inti sel ganas yang memang lebih besar dan lebih "
            "tidak beraturan.\n\n"
            "**Keputusan:** outlier **dipertahankan**. Menghapusnya justru membuang informasi penting untuk mengenali "
            "tumor ganas. Pengaruh nilai yang besar sudah diredam oleh standardisasi z-score."
        )

    # ------------------------------------------------------------ 5. Korelasi dengan target
    langkah(5, "Hubungan fitur dengan diagnosis", "Fitur mana yang paling membedakan Benign dan Malignant?")
    kor = sumber.eda_korelasi_target()
    rasio_kelas = sumber.eda_rasio_kelas()
    a, b = st.columns([1.3, 1])
    with a:
        st.altair_chart(grafik.korelasi_target(kor.rename(index=label_fitur), "Korelasi Setiap Fitur dengan Diagnosis"),
                        use_container_width=True)
    with b:
        n_kuat = int((kor.abs() >= 0.7).sum())
        atas = kor.head(5)
        bawah = kor.abs().sort_values().head(5)
        st.markdown(
            "Korelasi Pearson mengukur seberapa searah nilai fitur dengan diagnosis (Malignant = 1, Benign = 0). "
            "Nilai **mendekati 1** berarti makin besar nilai fitur, makin besar kemungkinan Malignant. Nilai "
            "**mendekati 0** berarti fitur tersebut hampir tidak membedakan kedua kelas.\n\n"
            f"Terdapat **{n_kuat} fitur** dengan korelasi kuat (|r| ≥ 0,7). Lima fitur paling berpengaruh:\n"
            + "\n".join(f"{i}. **{label_fitur(f)}** — r = {v:.3f}" for i, (f, v) in enumerate(atas.items(), 1))
            + "\n\nFitur yang berkaitan dengan **ukuran** (radius, perimeter, area) dan **bentuk cekung** (concave "
            "points, concavity) paling dominan, sesuai dengan ciri sel kanker yang membesar dan bentuknya tidak "
            "beraturan.\n\n"
            "Sebaliknya, fitur dengan korelasi paling lemah: "
            + ", ".join(f"*{label_fitur(f)}* ({kor[f]:+.3f})" for f in bawah.index)
            + ". Fitur-fitur ini tetap dipakai, karena model nonlinear seperti Random Forest dan SVM RBF masih bisa "
              "memanfaatkannya bersama fitur lain."
        )
    with st.expander("Lihat perbandingan rata-rata Malignant terhadap Benign"):
        st.markdown("Kolom **Rasio M/B** menunjukkan berapa kali rata-rata nilai fitur pasien Malignant dibandingkan "
                    "pasien Benign. Nilai jauh di atas 1 berarti fitur tersebut jauh lebih besar pada tumor ganas.")
        st.dataframe(pd.DataFrame({
            "Fitur": [label_fitur(f) for f in rasio_kelas.index],
            "Rata-rata Benign": rasio_kelas["rata_B"].values,
            "Rata-rata Malignant": rasio_kelas["rata_M"].values,
            "Rasio M/B": rasio_kelas["rasio_M_per_B"].values,
            "Korelasi (r)": kor.reindex(rasio_kelas.index).values,
        }).round(4), hide_index=True, use_container_width=True, height=420)

    # ------------------------------------------------------------ 6. Korelasi antarfitur
    langkah(6, "Hubungan antarfitur (multikolinearitas)", "Apakah ada fitur yang saling mirip informasinya?")
    pasangan = sumber.eda_pasangan_korelasi()
    pilih_var = st.radio("Variasi fitur pada heatmap", ["mean", "se", "worst"], horizontal=True,
                         format_func=lambda v: {"mean": "Mean", "se": "Standard error", "worst": "Worst"}[v],
                         key="variasi_heatmap")
    a, b = st.columns([1.3, 1])
    with a:
        st.altair_chart(grafik.heatmap_korelasi(sumber.eda_matriks_korelasi(pilih_var),
                                                f"Heatmap Korelasi Antarfitur ({pilih_var})"),
                        use_container_width=True)
    with b:
        st.markdown(
            f"Ditemukan **{len(pasangan)} pasangan fitur** dengan korelasi sangat tinggi (|r| > 0,9). Contoh paling "
            "jelas adalah **radius, perimeter, dan area**. Ketiganya sama-sama mengukur ukuran inti sel, sehingga "
            "secara geometris saling berkaitan (keliling ≈ 2πr, luas ≈ πr²).\n\n"
            "Warna **kuning** pada heatmap menunjukkan korelasi positif yang kuat, sedangkan warna terang berarti "
            "hampir tidak berkorelasi.\n\n"
            "**Keputusan:** seluruh **30 fitur tetap digunakan**, dengan alasan:\n"
            "- Random Forest memilih sebagian fitur secara acak di setiap node, sehingga fitur yang mirip tidak "
            "merusak model.\n"
            "- SVM RBF memakai semua fitur dalam perhitungan jarak.\n"
            "- SHAP perlu menilai kontribusi setiap fitur agar interpretasi model tetap lengkap."
        )
    with st.expander(f"Lihat {len(pasangan)} pasangan fitur dengan korelasi > 0,9"):
        st.dataframe(pd.DataFrame({
            "Fitur 1": pasangan["fitur_1"].map(label_fitur), "Fitur 2": pasangan["fitur_2"].map(label_fitur),
            "Korelasi": pasangan["korelasi"].round(3),
        }), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------ 7. Sebaran per fitur
    langkah(7, "Bandingkan sebaran fitur", "Lihat secara rinci perbedaan satu fitur antara Benign dan Malignant.")
    fitur = st.selectbox("Pilih fitur", FITUR, index=FITUR.index("concave_points_worst"), format_func=label_fitur)
    c, d = st.columns([1.4, 1])
    with c:
        st.altair_chart(grafik.histogram_per_kelas(df, fitur, label_fitur(fitur)), use_container_width=True)
    with d:
        ringkas = (df.groupby("label")[fitur].describe()[["mean", "50%", "std", "min", "max"]]
                   .rename(columns={"mean": "Rata-rata", "50%": "Median", "std": "Simpangan baku", "min": "Min",
                                    "max": "Maks"}).round(4))
        ringkas.index.name = "Kelas"
        st.dataframe(ringkas, use_container_width=True)
        st.caption(sumber.arti_fitur(fitur))
        r = float(kor[fitur])
        rb, rm = ringkas.loc["Benign", "Rata-rata"], ringkas.loc["Malignant", "Rata-rata"]
        bawah_irisan = max(ringkas.loc["Benign", "Min"], ringkas.loc["Malignant", "Min"])
        atas_irisan = min(ringkas.loc["Benign", "Maks"], ringkas.loc["Malignant", "Maks"])
        tumpang = int(df[fitur].between(bawah_irisan, atas_irisan).sum())
        tingkat = "kuat" if abs(r) >= 0.7 else "sedang" if abs(r) >= 0.4 else "lemah"
        arah = "lebih tinggi" if rm > rb else "lebih rendah"
        kali = f" (sekitar {rm / rb:.1f} kali)" if rb and rm > rb else ""
        catatan(
            f"<b>Interpretasi:</b> rata-rata <b>{label_fitur(fitur)}</b> pada pasien Malignant {arah} daripada "
            f"Benign{kali}. Korelasinya dengan diagnosis <b>{tingkat}</b> (r = {r:+.3f}). Sebanyak <b>{tumpang}</b> "
            f"dari {len(df)} pasien berada di daerah tumpang tindih kedua kelas, yaitu rentang nilai tempat Benign dan "
            "Malignant sama-sama muncul. Makin sedikit tumpang tindih, makin mudah fitur ini membedakan kedua kelas.",
            kuning=abs(r) >= 0.7,
        )

with t3:
    prep = sumber.hasil_preprocessing()
    if prep is None:
        st.warning("Catatan preprocessing belum tersedia (models/hasil_preprocessing.json).")
    else:
        st.markdown("Setiap langkah preprocessing ditentukan berdasarkan temuan EDA:")
        st.dataframe(pd.DataFrame(prep["rencana"]).rename(columns={
            "langkah": "Langkah", "temuan": "Temuan EDA", "tindakan": "Tindakan", "alasan": "Alasan"}),
            hide_index=True, use_container_width=True)
        catatan("Pada aplikasi, data pasien baru diproses dengan <b>scaler yang sama</b> dengan data latih "
                "(nilai μ dan σ dari data latih), sehingga tidak ada pelatihan ulang saat aplikasi dipakai.")

with t4:
    st.dataframe(df[["id", "diagnosis"] + FITUR], hide_index=True, use_container_width=True, height=420)
    st.download_button(":material/download: Unduh dataset (CSV)", df[["id", "diagnosis"] + FITUR].to_csv(index=False).encode("utf-8"),
                       "breast_cancer_wisconsin.csv", "text/csv")

kaki()
