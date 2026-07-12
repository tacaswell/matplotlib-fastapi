# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import mpl_sphinx_theme  # noqa: F401

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mpl-fastapi"
copyright = "2026, Thomas A Caswell"
author = "Thomas A Caswell"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_js",
    "numpydoc",  # Needs to be loaded *after* autodoc.
    "sphinxcontrib.openapi",
]
js_language = "typescript"
js_source_path = "../../mpl_fastapi/static/js/src"
templates_path = ["_templates"]
exclude_patterns = []

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True  # Generate stub files automatically

# -- Numpydoc configuration --------------------------------------------------
# Don't create a toctree for class members (which causes stub file warnings)
numpydoc_class_members_toctree = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "mpl_sphinx_theme"
html_static_path = ["_static"]
