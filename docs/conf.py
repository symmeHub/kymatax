import os
import sys
from datetime import datetime

# Ensure the package can be imported when building docs
sys.path.insert(0, os.path.abspath("../src"))

project = "kymatax"
author = "kymatax contributors"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# If you want to sync with the package version, keep in sync with pyproject
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"
html_static_path = ["_static"]

