"""Page d'affichage et de téléchargement des données transformées."""
import io

import streamlit as st

from datakit.preprocessing.utils.arrow_fix import safe_display_dataframe


def render_processed_page() -> None:
    st.markdown('<p class="sub-header">📊 Données Transformées</p>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.info("ℹ️ Aucune donnée transformée. Lancez le preprocessing.")
        return

    df = st.session_state.processed_data

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Lignes transformées", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes transformées", f"{len(df.columns)}")
    with col3:
        memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("💾 Mémoire", f"{memory:.1f} MB")

    st.dataframe(safe_display_dataframe(df), width="stretch", height=400)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger CSV", data=csv, file_name="processed_data.csv",
            mime="text/csv", use_container_width=True,
        )
    with col2:
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        st.download_button(
            "📥 Télécharger Parquet", data=parquet_buffer.getvalue(), file_name="processed_data.parquet",
            mime="application/octet-stream", use_container_width=True,
        )
