"""Barre latérale d'information sur l'état courant des données."""
import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 AI Experimentation Platform")
        st.markdown("---")

        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            st.markdown("### 📊 Données chargées")
            st.write(f"**Lignes:** {len(df):,}")
            st.write(f"**Colonnes:** {len(df.columns)}")

            if st.session_state.processed_data is not None:
                df_processed = st.session_state.processed_data
                st.markdown("### ✅ Données traitées")
                st.write(f"**Lignes:** {len(df_processed):,}")
                st.write(f"**Colonnes:** {len(df_processed.columns)}")

        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.caption("Version 1.0.0")
        st.caption("Made with ❤️ using Streamlit")
