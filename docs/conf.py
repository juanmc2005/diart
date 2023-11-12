# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'diart'
copyright = '2023, Juan Manuel Coria'
author = 'Juan Manuel Coria'
release = 'v0.9.0'

# sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

autoapi_dirs = ['../src/diart']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_title = 'Documentation'


def skip_submodules(app, what, name, obj, skip, options):
    return (
        name.startswith("_")
        or name.endswith("__init__")
        or name.startswith("diart.console")
        or name.startswith("diart.argdoc")
    )


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)
