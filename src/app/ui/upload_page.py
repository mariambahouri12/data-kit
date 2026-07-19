"""Page d'upload des données."""
import streamlit as st


def render_upload_page() -> None:
    st.markdown('<p class="sub-header">📤 Upload Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV, Excel ou Parquet",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Taille max: 100MB",
    )
    if uploaded_file is not None:
        _handle_file_upload(uploaded_file)


def _handle_file_upload(uploaded_file) -> None:
    try:
        df = st.session_state.file_loader.load(uploaded_file)
        st.session_state.current_data = df

        st.markdown(
            f'<div class="success-box">✅ Fichier chargé avec succès !<br>'
            f'<code>{uploaded_file.name}</code> — {len(df)} lignes, {len(df.columns)} colonnes</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")