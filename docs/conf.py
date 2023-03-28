import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../eir_auto_gp"))

"""Sphinx configuration."""
project = "EIR-auto-GP"
author = "Arnor Sigurdsson"
html_logo = "source/_static/img/eir-auto-gp-logo.svg"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
html_static_path = ["source/_static"]

html_css_files = [
    "css/custom.css",
]
copyright = f"{datetime.now().year}, {author}"
html_theme = "sphinx_rtd_theme"
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
]
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
