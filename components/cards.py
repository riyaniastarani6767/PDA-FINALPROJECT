import streamlit as st


def tampilkan_metric_card(label: str, value: str, key: str | None = None) -> None:
    """Menampilkan satu kartu metrik dengan markup visual konsisten."""
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-card__label">{label}</div>'
        f'<div class="metric-card__value">{value}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
