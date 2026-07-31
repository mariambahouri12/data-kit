# ui/sidebar.py
"""Barre latérale d'information sur l'état courant des données."""
import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📊 AI Experimentation Platform")
        st.markdown("---")

        # --- LLM Assistant Button ---
        st.markdown("### 🤖 Assistant IA")
        
        # Check if LLM is available
        llm_available = False
        if "ai_assistant" in st.session_state:
            llm_client = st.session_state.ai_assistant.get("llm_client")
            if llm_client:
                llm_available = llm_client.is_available
        
        # Display status indicator
        status_col, button_col = st.columns([0.3, 0.7])
        with status_col:
            if llm_available:
                st.markdown("🟢")
            else:
                st.markdown("🟡")
        with button_col:
            if st.button(
                "💬 Ouvrir l'assistant",
                use_container_width=True,
                type="primary" if llm_available else "secondary",
                key="llm_assistant_button"
            ):
                st.session_state.page = "llm_assistant"
                st.rerun()
        
        # Show status message
        if llm_available:
            st.caption("✅ LLM prêt (Mistral)")
        else:
            st.caption("⚠️ LLM non disponible")
            with st.expander("🔧 Configuration", expanded=False):
                st.info("""
                **Pour activer l'assistant :**
                1. Installez Ollama
                2. Lancez `ollama pull mistral`
                3. Redémarrez l'application
                """)
        
        st.markdown("---")

        # --- Data Status ---
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
            
            # --- Model Status ---
            if st.session_state.trained_model is not None:
                st.markdown("### 🤖 Modèle entraîné")
                st.write(f"**Modèle:** {st.session_state.model_name}")
                st.write(f"**Tâche:** {st.session_state.model_task}")
                
                if st.session_state.model_metrics:
                    # Afficher la meilleure métrique
                    metrics = st.session_state.model_metrics
                    # Filtrer les métriques None
                    valid_metrics = {k: v for k, v in metrics.items() if v is not None and isinstance(v, (int, float))}
                    if valid_metrics:
                        best_metric_name = max(valid_metrics.items(), key=lambda x: x[1])
                        st.write(f"**{best_metric_name[0]}:** {best_metric_name[1]:.3f}")

        st.markdown("---")
        st.markdown("### ℹ️ Info")
        st.caption("Version 1.0.0")
        st.caption("Made with ❤️ using Streamlit")