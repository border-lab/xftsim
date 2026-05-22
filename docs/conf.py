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
release = '0.3.0'

# -- Path setup --------------------------------------------------------------
# Add the project root so autodoc can import xftsim
sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#html_logo = "_static/xftsimlogomedium.svg"
html_logo = "_static/xftsimlogomediumwhite.svg"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

# Optionally enable extensions that may not be installed everywhere
_optional_extensions = [
    "sphinx_rtd_theme",
    "myst_parser",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]
for _ext in _optional_extensions:
    try:
        __import__(_ext.split(".")[0])
        extensions.append(_ext)
    except ImportError:
        pass

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints',
                    'plans', 'devtools', 'requirements.in', 'requirements.txt',
                    # Legacy orphaned documents not included in any toctree.
                    # These are pre-existing files from the old doc structure;
                    # excluding them avoids "document isn't included in any
                    # toctree" warnings without modifying the legacy content.
                    'api.md',
                    'examples.md',
                    'gettingstarted/getting_started.md',
                    'userguide/user_guide.md',
                    'userguide/submodules.md',
                    ]

# -- nbsphinx settings -------------------------------------------------------
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

# Support both reStructuredText and Markdown (if myst_parser is available)
source_suffix = {
    '.rst': 'restructuredtext',
}
if "myst_parser" in extensions:
    source_suffix['.md'] = 'markdown'
    myst_enable_extensions = ['attrs_inline', 'substitution']

autosectionlabel_prefix_document = True
autosummary_generate = True

# -- Mock imports for autodoc ------------------------------------------------
# nptyping is incompatible with newer numpy (removed np.bool8); mock it so
# autodoc can import the xftsim modules without the nptyping dependency.
autodoc_mock_imports = ['nptyping']

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Napoleon (NumPy docstrings) ---------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# -- sphinx-autodoc-typehints ------------------------------------------------
always_document_param_types = True
typehints_fully_qualified = False

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Use sphinx_rtd_theme if available, fall back to alabaster
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

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
