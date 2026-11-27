"""Grafik interaktif (Altair / Vega-Lite) dengan palet hijau-kuning yang sama dengan laporan.

Semua grafik dapat di-hover untuk melihat nilai (tooltip), diperbesar (tombol fullscreen),
dan diunduh sebagai PNG/SVG dari menu grafik. Ditampilkan dengan st.altair_chart(...).
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from tampilan.gaya import PALET, mode_gelap

CM_HIJAU, CM_HIJAU_MUDA, CM_KUNING = "#4A6F23", "#C0D1A6", "#F0C350"
FONT = "Source Sans Pro, sans-serif"


def _p() -> dict:
    return PALET["gelap" if mode_gelap() else "terang"]


def _warna_kelas() -> alt.Scale:
    return alt.Scale(domain=["Benign", "Malignant"], range=[_p()["hijau"], CM_KUNING])


def _gaya(chart, judul: str | None = None, tinggi: int = 320):
    """Menyeragamkan judul, font, warna sumbu, dan latar (mengikuti mode terang/gelap)."""
    p = _p()
    if judul:
        chart = chart.properties(title=alt.TitleParams(judul, anchor="start", fontSize=14, fontWeight=700,
                                                       color=p["hijau_tua"], font=FONT, offset=10))
    return (
        chart.properties(height=tinggi, background=p["kartu"], padding={"left": 8, "right": 12, "top": 8, "bottom": 6})
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=p["teks"], titleColor=p["teks"], gridColor=p["garis"], domainColor=p["garis"],
                        tickColor=p["garis"], labelFont=FONT, titleFont=FONT, labelFontSize=11, titleFontSize=11.5,
                        titleFontWeight=600)
        .configure_legend(labelColor=p["teks"], titleColor=p["teks"], labelFont=FONT, titleFont=FONT,
                          orient="bottom", labelFontSize=11.5)
        .configure_text(font=FONT)
    )


# ---------------------------------------------------------------------------
def confusion_matrix(TN: int, FP: int, FN: int, TP: int, judul: str):
    total = [TN + FP, FN + TP]
    data = pd.DataFrame([
        {"Aktual": "Benign", "Prediksi": "Benign", "n": TN, "kode": "TN", "warna": CM_HIJAU, "teks": "white",
         "arti": "Benign diprediksi Benign (benar)", "persen": TN / total[0] if total[0] else 0},
        {"Aktual": "Benign", "Prediksi": "Malignant", "n": FP, "kode": "FP", "warna": CM_HIJAU_MUDA, "teks": "#1F3D0C",
         "arti": "Benign keliru diprediksi Malignant", "persen": FP / total[0] if total[0] else 0},
        {"Aktual": "Malignant", "Prediksi": "Benign", "n": FN, "kode": "FN", "warna": CM_KUNING, "teks": "#1F3D0C",
         "arti": "Malignant keliru diprediksi Benign (paling berisiko)", "persen": FN / total[1] if total[1] else 0},
        {"Aktual": "Malignant", "Prediksi": "Malignant", "n": TP, "kode": "TP", "warna": CM_HIJAU, "teks": "white",
         "arti": "Malignant diprediksi Malignant (benar)", "persen": TP / total[1] if total[1] else 0},
    ])
    data["label"] = data.apply(lambda r: f"{r.kode} ({r.persen:.1%})".replace(".", ","), axis=1)
    x = alt.X("Prediksi:N", title="Kelas prediksi", sort=["Benign", "Malignant"], axis=alt.Axis(labelAngle=0))
    y = alt.Y("Aktual:N", title="Kelas aktual", sort=["Benign", "Malignant"])
    tip = [alt.Tooltip("kode:N", title="Sel"), alt.Tooltip("arti:N", title="Arti"), alt.Tooltip("n:Q", title="Jumlah"),
           alt.Tooltip("persen:Q", title="Persen dari kelas aktual", format=".1%")]
    dasar = alt.Chart(data).encode(x=x, y=y)
    kotak = dasar.mark_rect(cornerRadius=6, stroke=_p()["kartu"], strokeWidth=4).encode(
        color=alt.Color("warna:N", scale=None), tooltip=tip)
    angka = dasar.mark_text(fontSize=30, fontWeight="bold", dy=-10).encode(
        text="n:Q", color=alt.Color("teks:N", scale=None), tooltip=tip)
    ket = dasar.mark_text(fontSize=12, dy=20).encode(text="label:N", color=alt.Color("teks:N", scale=None), tooltip=tip)
    return _gaya((kotak + angka + ket).properties(width="container"), judul.replace("\n", " "), tinggi=330)


def shap_lokal(tabel: pd.DataFrame, top: int = 10, judul: str | None = None):
    """Batang kuning = mendorong ke Malignant, hijau = mendorong ke Benign (hover untuk detail)."""
    data = tabel.head(top).copy()
    data["Arah"] = np.where(data["Nilai SHAP"] > 0, "Mendorong ke Malignant", "Mendorong ke Benign")
    data["Label"] = [f"{f} = {v:.4g}" for f, v in zip(data["Fitur"], data["Nilai fitur"])]
    data["teks"] = [f"{v:+.4f}" for v in data["Nilai SHAP"]]
    urutan = data["Label"].tolist()
    warna = alt.Scale(domain=["Mendorong ke Malignant", "Mendorong ke Benign"], range=[CM_KUNING, _p()["hijau"]])
    tip = [alt.Tooltip("Fitur:N"), alt.Tooltip("Nilai fitur:Q", format=".5g"), alt.Tooltip("Nilai SHAP:Q", format="+.5f"),
           alt.Tooltip("Arah:N")]
    dasar = alt.Chart(data).encode(y=alt.Y("Label:N", sort=urutan, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)))
    batang = dasar.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Nilai SHAP:Q", title="Nilai SHAP (kontribusi terhadap probabilitas Malignant)"),
        color=alt.Color("Arah:N", scale=warna, title=None), tooltip=tip)
    teks_pos = dasar.transform_filter("datum['Nilai SHAP'] >= 0").mark_text(
        align="left", dx=4, fontSize=11, color=_p()["teks"]).encode(x="Nilai SHAP:Q", text="teks:N")
    teks_neg = dasar.transform_filter("datum['Nilai SHAP'] < 0").mark_text(
        align="right", dx=-4, fontSize=11, color=_p()["teks"]).encode(x="Nilai SHAP:Q", text="teks:N")
    nol = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_p()["abu"]).encode(x="x:Q")
    return _gaya((batang + teks_pos + teks_neg + nol).properties(width="container"),
                 judul or f"{len(data)} Fitur dengan Kontribusi Terbesar (SVM)", tinggi=34 * len(data) + 40)


def batang_horizontal(nama, nilai, judul: str, xlabel: str, warna: str | None = None, desimal: int = 4):
    data = pd.DataFrame({"Fitur": list(nama), "Nilai": list(nilai)})
    data["teks"] = data["Nilai"].map(lambda v: f"{v:.{desimal}f}")
    tip = [alt.Tooltip("Fitur:N"), alt.Tooltip("Nilai:Q", title=xlabel, format=f".{desimal + 1}f")]
    dasar = alt.Chart(data).encode(y=alt.Y("Fitur:N", sort=None, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)))
    batang = dasar.mark_bar(cornerRadiusEnd=4, color=warna or _p()["hijau"]).encode(
        x=alt.X("Nilai:Q", title=xlabel, scale=alt.Scale(domain=[0, float(data["Nilai"].max()) * 1.18])), tooltip=tip)
    teks = dasar.mark_text(align="left", dx=4, fontSize=11, color=_p()["teks"]).encode(x="Nilai:Q", text="teks:N", tooltip=tip)
    return _gaya((batang + teks).properties(width="container"), judul, tinggi=max(210, 32 * len(data) + 40))


def bandingan_metrik(tabel: pd.DataFrame, judul: str):
    """tabel: index = metrik, kolom = nama model, nilai dalam persen."""
    data = tabel.reset_index(names="Metrik").melt(id_vars="Metrik", var_name="Model", value_name="Nilai")
    data["teks"] = data["Nilai"].map(lambda v: f"{v:.2f}")
    urut = list(tabel.index)
    warna = alt.Scale(domain=list(tabel.columns), range=[_p()["hijau"], CM_KUNING])
    bawah = max(0, float(tabel.min().min()) - 6)
    tip = [alt.Tooltip("Model:N"), alt.Tooltip("Metrik:N"), alt.Tooltip("Nilai:Q", title="Nilai (%)", format=".2f")]
    dasar = alt.Chart(data).encode(x=alt.X("Metrik:N", sort=urut, title=None, axis=alt.Axis(labelAngle=0)),
                                   xOffset=alt.XOffset("Model:N"))
    batang = dasar.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, clip=True).encode(
        y=alt.Y("Nilai:Q", title="Nilai (%)", scale=alt.Scale(domain=[bawah, 100.5], zero=False)),
        y2=alt.datum(bawah), color=alt.Color("Model:N", scale=warna, title=None), tooltip=tip)
    teks = dasar.mark_text(dy=-6, fontSize=10.5, color=_p()["teks"]).encode(y="Nilai:Q", text="teks:N", tooltip=tip)
    return _gaya((batang + teks).properties(width="container"), judul, tinggi=330)


def boxplot_kfold(df: pd.DataFrame, metrik: str, judul: str):
    data = df.assign(Nilai=df[metrik] * 100, Model=df["model"],
                     Uji=df["ulangan"].astype(str) + "-" + df["fold"].astype(str))
    warna = alt.Scale(domain=list(data["Model"].unique()), range=[_p()["hijau"], CM_KUNING])
    kotak = alt.Chart(data).mark_boxplot(size=70, extent="min-max", median={"color": _p()["teks"]},
                                         rule={"color": _p()["abu"]}).encode(
        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Nilai:Q", title=f"{judul} (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("Model:N", scale=warna, legend=None))
    titik = alt.Chart(data).mark_circle(size=32, opacity=0.55, color=_p()["hijau_tua"]).encode(
        x="Model:N", y="Nilai:Q",
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Uji:N", title="Ulangan-fold"),
                 alt.Tooltip("Nilai:Q", title=f"{judul} (%)", format=".2f")])
    return _gaya((kotak + titik).properties(width="container"),
                 f"Sebaran {judul} — {len(data) // 2} pengujian per model", tinggi=330)


def distribusi_kelas(benign: int, malignant: int):
    data = pd.DataFrame({"Kelas": ["Benign", "Malignant"], "Jumlah": [benign, malignant]})
    total = benign + malignant
    data["teks"] = data["Jumlah"].map(lambda v: f"{v} ({v / total:.1%})")
    dasar = alt.Chart(data).encode(x=alt.X("Kelas:N", title=None, axis=alt.Axis(labelAngle=0)))
    batang = dasar.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=80).encode(
        y=alt.Y("Jumlah:Q", title="Jumlah pasien", scale=alt.Scale(domain=[0, max(benign, malignant) * 1.15])),
        color=alt.Color("Kelas:N", scale=_warna_kelas(), legend=None), tooltip=["Kelas", "Jumlah"])
    teks = dasar.mark_text(dy=-8, fontSize=12, color=_p()["teks"]).encode(y="Jumlah:Q", text="teks:N")
    return _gaya((batang + teks).properties(width="container"), "Distribusi Kelas Diagnosis", tinggi=300)


def histogram_per_kelas(df: pd.DataFrame, fitur: str, judul: str):
    data = pd.DataFrame({"Nilai": df[fitur], "Kelas": df["label"]})
    chart = alt.Chart(data).mark_bar(opacity=0.8, binSpacing=1).encode(
        x=alt.X("Nilai:Q", bin=alt.Bin(maxbins=30), title=judul),
        y=alt.Y("count():Q", title="Jumlah pasien", stack=None),
        color=alt.Color("Kelas:N", scale=_warna_kelas(), title=None),
        tooltip=[alt.Tooltip("Kelas:N"), alt.Tooltip("Nilai:Q", bin=alt.Bin(maxbins=30), title="Rentang nilai"),
                 alt.Tooltip("count():Q", title="Jumlah pasien")],
    )
    return _gaya(chart.properties(width="container"), f"Sebaran {judul} per Kelas", tinggi=300)


def shap_summary(matriks, X_asli, nama_fitur, top: int = 15):
    """SHAP summary (beeswarm) interaktif: titik = pasien, warna = nilai fitur (hijau rendah -> kuning tinggi)."""
    m = np.asarray(matriks, dtype=float)
    X = np.asarray(X_asli, dtype=float)
    urut = np.argsort(-np.abs(m).mean(axis=0))[:top]
    rng = np.random.default_rng(42)
    baris = []
    for j in urut:
        x = X[:, j]
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        skala = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
        for i in range(m.shape[0]):
            baris.append({"Fitur": nama_fitur[j], "SHAP": m[i, j], "Nilai fitur": x[i], "Skala": float(skala[i]),
                          "Pasien": i + 1, "jit": float(np.clip(rng.normal(0, 0.17), -0.45, 0.45))})
    data = pd.DataFrame(baris)
    urutan = [nama_fitur[j] for j in urut]
    titik = alt.Chart(data).mark_circle(size=24, opacity=0.85).encode(
        x=alt.X("SHAP:Q", title="Nilai SHAP (kontribusi terhadap probabilitas Malignant)"),
        y=alt.Y("Fitur:N", sort=urutan, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("jit:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
        color=alt.Color("Skala:Q", scale=alt.Scale(domain=[0, 1], range=[_p()["hijau"], "#B9C96B", CM_KUNING]),
                        title="Nilai fitur", legend=alt.Legend(orient="right", gradientLength=160,
                                                               labelExpr="datum.value == 0 ? 'rendah' : datum.value == 1 ? 'tinggi' : ''")),
        tooltip=[alt.Tooltip("Pasien:Q", title="Pasien uji ke-"), alt.Tooltip("Fitur:N"),
                 alt.Tooltip("Nilai fitur:Q", format=".5g"), alt.Tooltip("SHAP:Q", title="Nilai SHAP", format="+.5f")],
    )
    nol = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_p()["abu"]).encode(x="x:Q")
    return _gaya((nol + titik).properties(width="container"), "SHAP Summary Plot — SVM RBF (data uji)",
                 tinggi=32 * len(urutan) + 40)


def donat(nilai: dict, judul: str):
    """Diagram donat komposisi Benign/Malignant (hover untuk jumlah dan persentase)."""
    total = sum(nilai.values())
    if total == 0:
        data = pd.DataFrame({"Kelas": ["Belum ada data"], "Jumlah": [1], "Persen": [0.0]})
        skala = alt.Scale(domain=["Belum ada data"], range=[_p()["garis"]])
    else:
        data = pd.DataFrame({"Kelas": list(nilai.keys()), "Jumlah": list(nilai.values())})
        data["Persen"] = data["Jumlah"] / total
        skala = _warna_kelas()
    busur = alt.Chart(data).mark_arc(innerRadius=62, outerRadius=100, stroke=_p()["kartu"], strokeWidth=3).encode(
        theta=alt.Theta("Jumlah:Q", stack=True),
        color=alt.Color("Kelas:N", scale=skala, title=None),
        tooltip=[alt.Tooltip("Kelas:N"), alt.Tooltip("Jumlah:Q"), alt.Tooltip("Persen:Q", format=".1%")],
    )
    tengah = alt.Chart(pd.DataFrame({"t": [f"{total}"]})).mark_text(
        fontSize=26, fontWeight="bold", color=_p()["hijau_tua"], dy=-6).encode(text="t:N")
    sub = alt.Chart(pd.DataFrame({"t": ["total"]})).mark_text(fontSize=11, color=_p()["abu"], dy=16).encode(text="t:N")
    return _gaya((busur + tengah + sub).properties(width="container"), judul, tinggi=280)


def tren_harian(df: pd.DataFrame, judul: str):
    """df: kolom tanggal (date) dan hasil (Benign/Malignant). Batang bertumpuk per hari (14 hari terakhir)."""
    data = df.groupby(["tanggal", "hasil"]).size().rename("Jumlah").reset_index()
    hari = sorted(data["tanggal"].unique())[-14:]
    data = data[data["tanggal"].isin(hari)].copy()
    data["Tanggal"] = pd.to_datetime(data["tanggal"]).dt.strftime("%d/%m/%Y")
    urut = [pd.Timestamp(h).strftime("%d/%m/%Y") for h in hari]
    total = data.groupby("Tanggal")["Jumlah"].sum().reset_index()
    batang = alt.Chart(data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Tanggal:N", sort=urut, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Jumlah:Q", title="Jumlah prediksi", stack="zero",
                scale=alt.Scale(domain=[0, float(total["Jumlah"].max()) * 1.15])),
        color=alt.Color("hasil:N", scale=_warna_kelas(), title=None),
        tooltip=[alt.Tooltip("Tanggal:N"), alt.Tooltip("hasil:N", title="Hasil"), alt.Tooltip("Jumlah:Q")],
    )
    teks = alt.Chart(total).mark_text(dy=-8, fontSize=11, color=_p()["teks"]).encode(
        x=alt.X("Tanggal:N", sort=urut), y="Jumlah:Q", text="Jumlah:Q")
    return _gaya((batang + teks).properties(width="container"), judul, tinggi=300)


# ---------------------------------------------------------------------------
# Grafik tambahan untuk tab Eksplorasi Data
# ---------------------------------------------------------------------------
def skala_fitur(stat: pd.DataFrame, judul: str):
    """Nilai maksimum setiap fitur pada sumbu log untuk menunjukkan perbedaan skala."""
    data = pd.DataFrame({"Fitur": stat.index, "Maks": stat["max"].values, "Min": stat["min"].values,
                         "Rata": stat["mean"].values}).sort_values("Maks", ascending=False)
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3, color=_p()["hijau"]).encode(
        y=alt.Y("Fitur:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Maks:Q", title="Nilai maksimum (skala logaritmik)", scale=alt.Scale(type="log")),
        tooltip=[alt.Tooltip("Fitur:N"), alt.Tooltip("Min:Q", format=".5g"), alt.Tooltip("Rata:Q", title="Rata-rata", format=".5g"),
                 alt.Tooltip("Maks:Q", title="Maksimum", format=".5g")],
    )
    return _gaya(chart.properties(width="container"), judul, tinggi=22 * len(data) + 40)


def outlier_per_fitur(tabel: pd.DataFrame, judul: str, top: int = 12):
    """Batang bertumpuk jumlah outlier (IQR) per fitur, dipisah menurut kelas."""
    t = tabel.head(top)
    data = pd.concat([
        pd.DataFrame({"Fitur": t.index, "Kelas": "Malignant", "Jumlah": t["dari_malignant"].values}),
        pd.DataFrame({"Fitur": t.index, "Kelas": "Benign", "Jumlah": t["dari_benign"].values}),
    ])
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=2).encode(
        y=alt.Y("Fitur:N", sort=list(t.index), title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Jumlah:Q", title="Jumlah nilai outlier", stack="zero"),
        color=alt.Color("Kelas:N", scale=_warna_kelas(), title=None),
        order=alt.Order("Kelas:N", sort="descending"),
        tooltip=[alt.Tooltip("Fitur:N"), alt.Tooltip("Kelas:N"), alt.Tooltip("Jumlah:Q", title="Jumlah outlier")],
    )
    return _gaya(chart.properties(width="container"), judul, tinggi=26 * len(t) + 60)


def korelasi_target(kor: pd.Series, judul: str):
    """Korelasi Pearson setiap fitur terhadap diagnosis (M = 1, B = 0); kuning = kuat, hijau = lemah."""
    data = pd.DataFrame({"Fitur": kor.index, "r": kor.values})
    data["Kekuatan"] = np.where(data["r"].abs() >= 0.7, "Kuat (|r| ≥ 0,7)",
                                np.where(data["r"].abs() >= 0.4, "Sedang (0,4 – 0,7)", "Lemah (< 0,4)"))
    warna = alt.Scale(domain=["Kuat (|r| ≥ 0,7)", "Sedang (0,4 – 0,7)", "Lemah (< 0,4)"],
                      range=[CM_KUNING, _p()["hijau"], _p()["hijau_muda"]])
    batang = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Fitur:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("r:Q", title="Korelasi Pearson dengan diagnosis (Malignant = 1)", scale=alt.Scale(domain=[-0.2, 1])),
        color=alt.Color("Kekuatan:N", scale=warna, title=None),
        tooltip=[alt.Tooltip("Fitur:N"), alt.Tooltip("r:Q", title="Korelasi (r)", format="+.3f"), alt.Tooltip("Kekuatan:N")],
    )
    nol = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=_p()["abu"]).encode(x="x:Q")
    return _gaya((nol + batang).properties(width="container"), judul, tinggi=22 * len(data) + 60)


def heatmap_korelasi(kor: pd.DataFrame, judul: str):
    """Heatmap korelasi antarfitur (nilai absolut tidak dipakai agar arah tetap terlihat)."""
    nama = list(kor.columns)
    data = kor.reset_index(names="Fitur 1").melt(id_vars="Fitur 1", var_name="Fitur 2", value_name="r")
    data["teks"] = data["r"].map(lambda v: f"{v:.2f}")
    dasar = alt.Chart(data).encode(
        x=alt.X("Fitur 2:N", sort=nama, title=None, axis=alt.Axis(labelAngle=-45, labelLimit=160)),
        y=alt.Y("Fitur 1:N", sort=nama, title=None, axis=alt.Axis(labelLimit=160)),
    )
    kotak = dasar.mark_rect().encode(
        color=alt.Color("r:Q", title="r", scale=alt.Scale(domain=[-1, 0, 1], range=["#7A8F6A", _p()["kartu"], CM_KUNING])),
        tooltip=[alt.Tooltip("Fitur 1:N"), alt.Tooltip("Fitur 2:N"), alt.Tooltip("r:Q", title="Korelasi", format="+.3f")],
    )
    teks = dasar.mark_text(fontSize=10, color=_p()["teks"]).encode(text="teks:N")
    return _gaya((kotak + teks).properties(width="container"), judul, tinggi=30 * len(nama) + 80)
