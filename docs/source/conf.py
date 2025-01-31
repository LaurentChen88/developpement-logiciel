import os
import sys

sys.path.insert(
    0, os.path.abspath("../../src")
)  # Ajuste selon ton arborescence


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "developpement-logiciel"
copyright = "2025, Laurent Chen"
author = "Laurent Chen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Génération automatique de documentation depuis les docstrings
    "sphinx.ext.napoleon",  # Support du style Google et NumPy pour les docstrings
    "sphinx.ext.viewcode",  # Ajoute les liens vers le code source
    "sphinx.ext.todo",  # Permet d'utiliser des TODOs dans la doc
    "sphinx.ext.autosummary",  # Génère automatiquement des résumés de module
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Activer le support des docstrings Google
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
