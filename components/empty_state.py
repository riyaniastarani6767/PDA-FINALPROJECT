import streamlit as st


def tampilkan_empty_state(title: str, description: str, icon: str = "📭") -> None:
    """Menampilkan kondisi kosong yang konsisten lintas halaman."""
    st.markdown(
        f'<div class="empty-card"><div class="empty-card__icon">{icon}</div>'
        f'<div class="empty-card__title">{title}</div>'
        f'<div class="empty-card__description">{description}</div></div>',
        unsafe_allow_html=True,
    )
