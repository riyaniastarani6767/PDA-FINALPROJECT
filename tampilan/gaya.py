"""Tema warna dan komponen tampilan bersama.

Palet hijau-kuning mengikuti grafik pada laporan (confusion matrix, SHAP, feature importance):
- Hijau  : kelas Benign, elemen utama, tombol.
- Kuning : kelas Malignant, penekanan/peringatan.
Mode terang adalah bawaan; mode gelap dapat diaktifkan dari sidebar.
"""
from __future__ import annotations

import html

import streamlit as st

KUNCI_GELAP = "mode_gelap"

# Palet dasar (dipakai juga oleh grafik interaktif Altair di tampilan/grafik.py)
PALET = {
    "terang": {
        "hijau": "#4A6F23",
        "hijau_tua": "#1F3D0C",
        "hijau_muda": "#C0D1A6",
        "hijau_pucat": "#EEF3E6",
        "kuning": "#F0C350",
        "kuning_tua": "#9A6B00",
        "kuning_pucat": "#FDF4DC",
        "teks": "#1F2A1A",
        "abu": "#66705E",
        "latar": "#F8FAF4",
        "kartu": "#FFFFFF",
        "garis": "#DCE5CF",
    },
    "gelap": {
        "hijau": "#8DB35E",
        "hijau_tua": "#CFE3B4",
        "hijau_muda": "#3E5A27",
        "hijau_pucat": "#1C2817",
        "kuning": "#F0C350",
        "kuning_tua": "#F5D57E",
        "kuning_pucat": "#3A3016",
        "teks": "#E7EEDF",
        "abu": "#A4B09A",
        "latar": "#11180D",
        "kartu": "#182214",
        "garis": "#2F3E27",
    },
}

TEMA_STREAMLIT = {
    "terang": {"base": "light", "primaryColor": "#4A6F23", "backgroundColor": "#F8FAF4",
               "secondaryBackgroundColor": "#EEF3E6", "textColor": "#1F2A1A"},
    "gelap": {"base": "dark", "primaryColor": "#5E8A30", "backgroundColor": "#11180D",
              "secondaryBackgroundColor": "#1C2817", "textColor": "#E7EEDF"},
}


def mode_gelap() -> bool:
    try:
        return bool(st.session_state.get(KUNCI_GELAP, False))
    except Exception:
        return False


def warna() -> dict:
    return PALET["gelap" if mode_gelap() else "terang"]


def _css(p: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --hijau: {p['hijau']}; --hijau-tua: {p['hijau_tua']}; --hijau-muda: {p['hijau_muda']};
  --hijau-pucat: {p['hijau_pucat']}; --kuning: {p['kuning']}; --kuning-tua: {p['kuning_tua']};
  --kuning-pucat: {p['kuning_pucat']}; --teks: {p['teks']}; --abu: {p['abu']};
  --latar: {p['latar']}; --kartu: {p['kartu']}; --garis: {p['garis']};
  --bayang: 0 1px 2px rgba(16, 24, 12, .05), 0 1px 3px rgba(16, 24, 12, .06);
}}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
.stApp {{ background: var(--latar); }}
[data-testid="stSidebar"] {{ background: var(--kartu); border-right: 1px solid var(--garis); }}
[data-testid="stSidebarNav"] a {{ border-radius: 8px; font-weight: 500; }}
.block-container {{ padding-top: 3rem; max-width: 1200px; }}
h1, h2, h3 {{ color: var(--hijau-tua); font-weight: 650; letter-spacing: -.01em; }}
h3 {{ font-size: 1.15rem; margin-top: .6rem; }}

/* Identitas di sidebar */
.merek {{ display: flex; align-items: center; gap: .7rem; padding: .2rem 0 .9rem 0;
  border-bottom: 1px solid var(--garis); margin-bottom: .4rem; }}
.merek-logo {{ width: 2.3rem; height: 2.3rem; border-radius: 10px; background: var(--hijau); color: #fff;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
.merek-nama {{ font-weight: 650; font-size: .95rem; color: var(--teks); line-height: 1.25; }}
.merek-sub {{ font-size: .75rem; color: var(--abu); margin-top: .1rem; }}

/* Kepala halaman */
.kepala {{ background: var(--kartu); border: 1px solid var(--garis); border-left: 4px solid var(--hijau);
  border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.4rem; box-shadow: var(--bayang); }}
.kepala h1 {{ color: var(--hijau-tua); font-size: 1.6rem; font-weight: 700; margin: 0 0 .3rem 0; padding: 0; }}
.kepala p {{ color: var(--abu); margin: 0; font-size: .95rem; line-height: 1.55; }}
.kepala .lencana {{ display: inline-block; background: var(--hijau-pucat); color: var(--hijau);
  border: 1px solid var(--hijau-muda); font-weight: 600; font-size: .68rem; letter-spacing: .08em;
  padding: .15rem .55rem; border-radius: 6px; margin-bottom: .6rem; }}

/* Kartu umum */
.kartu {{ background: var(--kartu); border: 1px solid var(--garis); border-radius: 12px; padding: 1rem 1.15rem;
  height: 100%; box-shadow: var(--bayang); }}
.kartu h4 {{ margin: 0 0 .35rem 0; color: var(--hijau-tua); font-size: 1rem; font-weight: 650; }}
.kartu p {{ margin: 0; color: var(--abu); font-size: .88rem; line-height: 1.5; }}
.kartu-angka {{ background: var(--kartu); border: 1px solid var(--garis); border-top: 3px solid var(--hijau);
  border-radius: 12px; padding: .9rem 1.05rem; box-shadow: var(--bayang); height: 100%; }}
.kartu-angka.kuning {{ border-top-color: var(--kuning); }}
.kartu-angka .label {{ color: var(--abu); font-size: .74rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .05em; }}
.kartu-angka .nilai {{ color: var(--teks); font-size: 1.7rem; font-weight: 700; line-height: 1.25; margin-top: .2rem;
  font-variant-numeric: tabular-nums; }}
.kartu-angka .ket {{ color: var(--abu); font-size: .78rem; margin-top: .1rem; }}

/* Langkah bernomor */
.langkah {{ display: flex; align-items: center; gap: .75rem; margin: 1.6rem 0 .7rem 0; padding-bottom: .6rem;
  border-bottom: 1px solid var(--garis); }}
.langkah .no {{ background: var(--hijau); color: #fff; width: 1.75rem; height: 1.75rem; border-radius: 8px;
  display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: .88rem; flex-shrink: 0; }}
.langkah .judul {{ font-size: 1.05rem; font-weight: 650; color: var(--teks); }}
.langkah .sub {{ color: var(--abu); font-size: .84rem; }}

/* Kartu hasil prediksi */
.hasil {{ background: var(--kartu); border-radius: 12px; padding: 1.1rem 1.25rem; border: 1px solid var(--garis);
  border-top: 4px solid; height: 100%; box-shadow: var(--bayang); }}
.hasil.benign {{ border-top-color: var(--hijau); }}
.hasil.malignant {{ border-top-color: var(--kuning); }}
.hasil .model {{ font-size: .74rem; color: var(--abu); font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }}
.hasil .kelas {{ display: flex; align-items: center; gap: .45rem; font-size: 1.7rem; font-weight: 700; margin: .25rem 0; }}
.hasil .kelas svg {{ flex-shrink: 0; }}
.hasil.benign .kelas {{ color: var(--hijau); }}
.hasil.malignant .kelas {{ color: var(--kuning-tua); }}
.hasil .arti {{ color: var(--teks); font-size: .88rem; line-height: 1.5; margin-bottom: .8rem; }}
.bar {{ display: flex; height: 22px; border-radius: 6px; overflow: hidden; background: var(--hijau-pucat); }}
.bar div {{ display: flex; align-items: center; justify-content: center; font-size: .74rem; font-weight: 600;
  white-space: nowrap; overflow: hidden; }}
.bar .b {{ background: var(--hijau); color: #fff; }}
.bar .m {{ background: var(--kuning); color: #2B2000; }}
.bar-ket {{ display: flex; justify-content: space-between; font-size: .76rem; color: var(--abu); margin-top: .3rem; }}

/* Kotak info */
.catatan {{ background: var(--hijau-pucat); border: 1px solid var(--hijau-muda); border-left: 3px solid var(--hijau);
  border-radius: 8px; padding: .75rem 1rem; color: var(--teks); font-size: .88rem; line-height: 1.55; margin: .4rem 0 .9rem 0; }}
.catatan.kuning {{ background: var(--kuning-pucat); border-color: var(--kuning); border-left-color: var(--kuning-tua); }}
.kaki {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; color: var(--abu); font-size: .78rem;
  border-top: 1px solid var(--garis); padding-top: .9rem; margin-top: 2.2rem; }}

[data-testid="stMetric"] {{ background: var(--kartu); border: 1px solid var(--garis); border-radius: 12px;
  padding: .8rem 1rem; box-shadow: var(--bayang); }}
[data-testid="stMetricLabel"] p {{ font-size: .78rem; font-weight: 600; color: var(--abu); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--garis); border-radius: 10px; overflow: hidden; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--garis); border-radius: 10px; background: var(--kartu); }}
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid var(--garis); }}
.stTabs [data-baseweb="tab"] {{ font-weight: 500; padding: .5rem .85rem; }}
.stTabs [aria-selected="true"] {{ font-weight: 600; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 8px; font-weight: 500; }}
[data-testid="stPageLink"] a {{ background: var(--kartu); border: 1px solid var(--garis); border-radius: 10px;
  padding: .55rem .8rem; box-shadow: var(--bayang); }}
[data-testid="stPageLink"] a:hover {{ border-color: var(--hijau); }}
/* Warna kotak pesan Streamlit mengikuti palet hijau-kuning */
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background: var(--hijau-pucat); border-left: 3px solid var(--hijau); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background: var(--hijau-pucat); border-left: 3px solid var(--hijau); }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background: var(--kuning-pucat); border-left: 3px solid var(--kuning); }}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"], [data-testid="stAlertContentWarning"],
[data-testid="stAlertContentInfo"] p, [data-testid="stAlertContentSuccess"] p, [data-testid="stAlertContentWarning"] p {{ color: var(--teks); }}
</style>
"""


def _ubah_mode() -> None:
    st.session_state[KUNCI_GELAP] = st.session_state["_toggle_gelap"]


def terapkan_tema() -> None:
    """Toggle mode gelap di sidebar + opsi tema Streamlit + CSS. Dipanggil sekali dari app.py."""
    if KUNCI_GELAP not in st.session_state:
        st.session_state[KUNCI_GELAP] = False
    with st.sidebar:
        st.toggle(":material/dark_mode: Mode gelap", value=st.session_state[KUNCI_GELAP], key="_toggle_gelap",
                  on_change=_ubah_mode, help="Ganti tampilan antara mode terang dan mode gelap.")
    nama = "gelap" if st.session_state[KUNCI_GELAP] else "terang"
    berubah = False
    for opsi, nilai in TEMA_STREAMLIT[nama].items():
        if st.get_option(f"theme.{opsi}") != nilai:
            st._config.set_option(f"theme.{opsi}", nilai)
            berubah = True
    if berubah:
        st.rerun()
    # Baris kosong dihapus agar parser Markdown tidak memutus blok <style> dan menampilkan CSS sebagai teks.
    css = "\n".join(baris for baris in _css(PALET[nama]).splitlines() if baris.strip())
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Komponen
# ---------------------------------------------------------------------------
def e(teks) -> str:
    return html.escape(str(teks))


def kepala(judul: str, deskripsi: str, lencana: str | None = None) -> None:
    lenc = f'<div class="lencana">{e(lencana)}</div>' if lencana else ""
    st.markdown(f'<div class="kepala">{lenc}<h1>{e(judul)}</h1><p>{deskripsi}</p></div>', unsafe_allow_html=True)


def langkah(nomor: int | str, judul: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="langkah"><div class="no">{e(nomor)}</div><div><div class="judul">{e(judul)}</div>{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def kartu(judul: str, isi: str) -> None:
    st.markdown(f'<div class="kartu"><h4>{judul}</h4><p>{isi}</p></div>', unsafe_allow_html=True)


def kartu_angka(label: str, nilai: str, ket: str = "", kuning: bool = False) -> None:
    kelas = "kartu-angka kuning" if kuning else "kartu-angka"
    ket_html = f'<div class="ket">{ket}</div>' if ket else ""
    st.markdown(f'<div class="{kelas}"><div class="label">{e(label)}</div><div class="nilai">{e(nilai)}</div>{ket_html}</div>',
                unsafe_allow_html=True)


def catatan(teks: str, kuning: bool = False) -> None:
    st.markdown(f'<div class="catatan{" kuning" if kuning else ""}">{teks}</div>', unsafe_allow_html=True)


def bar_probabilitas(p_benign: float, p_malignant: float) -> str:
    pb, pm = max(p_benign, 0.0) * 100, max(p_malignant, 0.0) * 100
    tb = f"{pb:.1f}%" if pb >= 12 else ""
    tm = f"{pm:.1f}%" if pm >= 12 else ""
    return (f'<div class="bar"><div class="b" style="width:{pb:.2f}%">{tb}</div>'
            f'<div class="m" style="width:{pm:.2f}%">{tm}</div></div>'
            f'<div class="bar-ket"><span>Benign {pb:.1f}%</span><span>Malignant {pm:.1f}%</span></div>')


_SVG = ('<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round">{}</svg>')
_IKON_BENIGN = _SVG.format('<circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/>')
_IKON_MALIGNANT = _SVG.format('<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
                              '<path d="M12 9v4"/><path d="M12 17h.01"/>')


def kartu_hasil(nama_model: str, label: str, p_benign: float, p_malignant: float) -> None:
    jinak = label == "Benign"
    kelas = "benign" if jinak else "malignant"
    arti = ("Tumor diprediksi <b>jinak</b> (bukan kanker)." if jinak
            else "Tumor diprediksi <b>ganas</b> (kanker). Perlu pemeriksaan lanjutan oleh dokter.")
    yakin = max(p_benign, p_malignant)
    st.markdown(
        f'<div class="hasil {kelas}"><div class="model">{e(nama_model)}</div>'
        f'<div class="kelas">{_IKON_BENIGN if jinak else _IKON_MALIGNANT}{e(label)}</div>'
        f'<div class="arti">{arti} Keyakinan model: <b>{yakin:.1%}</b>.</div>'
        f'{bar_probabilitas(p_benign, p_malignant)}</div>',
        unsafe_allow_html=True,
    )


def kaki() -> None:
    st.markdown(
        '<div class="kaki"><span>Sistem Klasifikasi Tumor Payudara</span>'
        '<span>Random Forest &amp; Support Vector Machine · Feature Importance &amp; SHAP</span></div>',
        unsafe_allow_html=True,
    )


def tautan(halaman: str, label: str, ikon: str | None = None) -> None:
    """st.page_link yang aman ketika halaman dijalankan sendiri (mis. saat pengujian)."""
    try:
        st.page_link(halaman, label=label, icon=ikon)
    except Exception:
        st.markdown(f"{ikon or ''} {label}")
