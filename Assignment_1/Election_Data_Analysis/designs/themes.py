import importlib
import streamlit as st
from pathlib import Path


def get_theme_names():
    """Get list of available themes."""
    return ["Default", "Modern Dark", "Light Professional", "Vibrant Gradient", "Minimal"]


def get_theme_module(theme_name):
    """Import and return theme module based on theme name."""
    if theme_name == "Default":
        return None

    # Convert theme name to module name
    module_name = theme_name.lower().replace(" ", "_")
    try:
        return importlib.import_module(f"designs.styles.{module_name}")
    except ImportError:
        st.error(f"Theme {theme_name} not found!")
        return None


def apply_theme(theme_name):
    """Apply selected theme."""
    theme_module = get_theme_module(theme_name)
    if theme_module:
        st.markdown(theme_module.get_style(), unsafe_allow_html=True)