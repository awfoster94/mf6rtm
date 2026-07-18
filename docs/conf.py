# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'mf6rtm'
copyright = f'{datetime.now().year}, Pablo Ortega'
# year = datetime.now().year
author = 'Pablo Ortega, Anthony Aufdenkampe and others'
# release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",              # automatic API docs
    "sphinx.ext.napoleon",             # Google/NumPy docstrings
    "sphinx_autodoc_typehints",        # types from function annotations
    "myst_nb",                         # Markdown + notebook support
    "sphinx.ext.autosummary",
]

# Auto-generate summary tables for modules
autosummary_generate = True

nb_execution_mode = "off"

# On Read the Docs, pull the CI-executed tutorial notebooks via rtds-action.
on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    extensions.append("rtds_action")
    rtds_action_github_repo = "p-ortega/mf6rtm"
    rtds_action_path = "tutorials"
    rtds_action_artifact_prefix = "notebooks-for-"
    rtds_action_github_token = os.environ.get("GITHUB_TOKEN", None)
    rtds_action_error_if_missing = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "navigation_with_keys": True,   # optional: navigate with keyboard
}
