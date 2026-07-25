# ui/__init__.py
from .upload_page import render_upload_page
from .preview_page import render_preview_page
from .preprocessing_page import render_preprocessing_page
from .processed_page import render_processed_page
from .visualization_page import render_visualization_page
from .models_page import render_models_page  # NOUVEAU

__all__ = [
    "render_upload_page",
    "render_preview_page",
    "render_preprocessing_page",
    "render_processed_page",
    "render_visualization_page",
    "render_models_page",  # NOUVEAU
]