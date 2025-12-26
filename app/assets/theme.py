"""Theme and styling utilities for Streamlit app.

Author: João Pedro Cunha
"""

from pathlib import Path

import streamlit as st


def load_css():
    """Load custom CSS styles."""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def set_page_config(title: str, icon: str = "🏎️", layout: str = "wide"):
    """Configure Streamlit page with consistent settings.

    Args:
        title: Page title
        icon: Page icon (optional)
        layout: Page layout ("wide" or "centered")
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )
    load_css()


def create_header(title: str, subtitle: str = None):
    """Create consistent page header.

    Args:
        title: Main title
        subtitle: Optional subtitle
    """
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)


def create_data_status_panel(session_info: dict):
    """Create data status panel showing session information.

    Args:
        session_info: Dictionary with session metadata
    """
    st.markdown('<div class="data-status">', unsafe_allow_html=True)
    st.markdown("<h4>Data Status</h4>", unsafe_allow_html=True)

    for key, value in session_info.items():
        st.markdown(f"<p><strong>{key}:</strong> {value}</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
