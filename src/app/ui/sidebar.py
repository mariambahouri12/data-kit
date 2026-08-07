"""Barre latérale d'information sur l'état courant des données."""
import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 AI Experimentation Platform")
        st.markdown("---")

        # === DATA STATUS ===
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            st.markdown("### 📤 Données chargées")
            st.write(f"**Lignes:** {len(df):,}")
            st.write(f"**Colonnes:** {len(df.columns)}")

            if st.session_state.processed_data is not None:
                df_processed = st.session_state.processed_data
                st.markdown("### ✅ Données traitées")
                st.write(f"**Lignes:** {len(df_processed):,}")
                st.write(f"**Colonnes:** {len(df_processed.columns)}")
            
            # === MODEL STATUS ===
            if st.session_state.trained_model is not None:
                st.markdown("### 🤖 Modèle entraîné")
                st.write(f"**Modèle:** {st.session_state.model_name}")
                st.write(f"**Tâche:** {st.session_state.model_task}")
                
                if st.session_state.model_metrics:
                    valid_metrics = {k: v for k, v in st.session_state.model_metrics.items() 
                                   if v is not None and isinstance(v, (int, float))}
                    if valid_metrics:
                        best_metric_name = max(valid_metrics.items(), key=lambda x: x[1])
                        st.write(f"**{best_metric_name[0]}:** {best_metric_name[1]:.3f}")

        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.caption("Version 1.0.0")
        st.caption("Made with ❤️ using Streamlit")