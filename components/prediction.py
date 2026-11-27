import streamlit as st


def tampilkan_status_backend(status: str, pesan: str) -> None:
    if status == "READY":
        st.success(pesan)
    elif status == "MODEL_NOT_FOUND":
        st.warning(pesan)
    elif status == "MODEL_NOT_VERIFIED":
        st.info(pesan)
    else:
        st.error(pesan)


def tampilkan_area_hasil(respons: dict | None) -> None:
    """Menampilkan kontrak hasil tanpa membuat prediksi atau probabilitas palsu."""
    if respons is None:
        st.markdown(
            '<div class="empty-card"><div class="empty-card__icon">🧪</div>'
            '<div class="empty-card__title">Hasil prediksi belum tersedia</div>'
            '<div class="empty-card__description">Lengkapi input dan jalankan tombol prediksi.</div></div>',
            unsafe_allow_html=True,
        )
        return

    status = respons.get("status")
    if hasattr(status, "value"):
        status = status.value

    if status == "success":
        st.success(f"Prediksi berhasil menggunakan {respons.get('model')}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Hasil", respons.get("classification_label", "—"))
        col2.metric("Probabilitas Benign", f"{float(respons.get('probability_benign', 0.0)):.2%}")
        col3.metric("Probabilitas Malignant", f"{float(respons.get('probability_malignant', 0.0)):.2%}")
        if "decision_score" in respons:
            st.caption(f"SVM decision score: {respons['decision_score']:.6f}")
        st.caption(f"Metode input: {respons.get('input_method', '—')} | Status: {respons.get('model_status', '—')}")
        return

    if status in ("partial_success", "error") and "detail" in respons:
        st.write(f"Status batch: **{status}** — {respons.get('jumlah_berhasil', 0)} berhasil, {respons.get('jumlah_gagal', 0)} gagal")
        if respons.get("error"):
            st.warning(respons["error"])
        rows = []
        for detail in respons["detail"]:
            rows.append({
                "Baris": detail.get("row_number"),
                "Source index": detail.get("source_index"),
                "Status": detail.get("status"),
                "Hasil": detail.get("classification_label"),
                "Benign": None if detail.get("probability_benign") is None else f"{detail['probability_benign']:.2%}",
                "Malignant": None if detail.get("probability_malignant") is None else f"{detail['probability_malignant']:.2%}",
                "Decision score": detail.get("decision_score"),
                "Error": detail.get("error"),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
        return

    pesan = respons.get("error") or respons.get("pesan") or "Inference gagal dijalankan."
    if status == "MODEL_NOT_FOUND":
        st.warning(pesan)
    elif status == "MODEL_NOT_VERIFIED":
        st.info(pesan)
    else:
        st.error(pesan)
