"""Page d'upload des données."""
import streamlit as st

from datakit.exceptions import EmptyFileError


def render_upload_page() -> None:
    st.markdown('<p class="sub-header">📤 Upload Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file in CSV, Excel, or Parquet format",
        type=["csv", "xlsx", "xls", "parquet"],
        help="Max size: 100MB",
    )
    if uploaded_file is not None:
        _handle_file_upload(uploaded_file)


def _handle_file_upload(uploaded_file) -> None:
    try:
        df = st.session_state.file_loader.load(uploaded_file)
        st.session_state.current_data = df

        st.markdown(
            f'<div class="success-box">✅ File uploaded successfully!<br>'
            f'<code>{uploaded_file.name}</code> — {len(df)} rows, {len(df.columns)} columns</div>',
            unsafe_allow_html=True,
        )
    except EmptyFileError:
        st.error("The file is empty. Please choose another file.")

    except ValueError as e:
        st.error(str(e))

    except Exception:
        st.error("An unexpected error occurred while loading the file.")