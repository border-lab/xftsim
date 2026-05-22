import sys
import os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xftsim'
copyright = '2024-2026, Richard Border'
author = 'Richard Border'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath(".."))


#html_logo = "_static/xftsimlogomedium.svg"
html_logo = "_static/xftsimlogomediumwhite.svg"
#html_logo = "_static/xftsimlogo.svg"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "myst_parser",
    #"sphinx_autosummary_accessors",
    #"sphinxawesome_theme",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
#    "sphinx_autosummary_accessors",
#    "sphinx.ext.linkcode",
#    "sphinx_copybutton",
    #"sphinxext.rediraffe",
    #"sphinx_design",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = ['attrs_inline', 'substitution']

autosectionlabel_prefix_document = True
autosummary_generate = True
autodoc_typehints = "none"


# Napoleon configurations

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_theme='press'
html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinx_book_theme'
#html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']
html_theme_options = {
    'logo_only': True,
}
html_css_files = ['custom.css']

# Add "Edit on GitHub" links to the sphinx_rtd_theme sidebar.
# Read the Docs builds multiple branches (main, dev, v0.9alpha); use the branch
# it is currently building so edit links point at the right ref. Falls back to
# 'main' for local builds.
html_context = {
    'display_github': True,
    'github_user': 'border-lab',
    'github_repo': 'xftsim',
    'github_version': os.environ.get('READTHEDOCS_GIT_IDENTIFIER', 'main'),
    'conf_py_path': '/docs/',
}
