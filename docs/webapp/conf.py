# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))


# -- Project information -----------------------------------------------------

project = 'marlin'
copyright = '2021, marlindev'
author = 'marlindev'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    #'sphinx_autodoc_typehints',
    'recommonmark',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.imgmath',
    'sphinx.ext.autosectionlabel',
    # 'sphinx_copybutton',
    # 'sphinx_paramlinks',
    # 'sphinx_togglebutton',
]

napoleon_google_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# pip install git+https://github.com/bashtage/sphinx-material.git
# html_theme = 'sphinx_material'

# # Material theme options (see theme.conf for more information)
# # learn more: https://bashtage.github.io/sphinx-material/#getting-started
# html_sidebars = {
#     "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
# }
# html_theme_options = {

#     # Set the name of the project to appear in the navigation.
#     'nav_title': 'marlin',

#     # Specify a base_url used to generate sitemap.xml. If not
#     # specified, then no sitemap will be built.
#     'base_url': 'https://project.github.io/project',

    

#     # Set the color and the accent color
#     'color_primary': 'blue',
#     'color_accent': 'light-blue',

#     # Set the repo location to get a badge with stats
#     'repo_url': 'https://o365exchange.visualstudio.com/O365%20Core/_git/ELR?path=%2Fsources%2Fdev%2FSubstrateInferences%2Fmarlin&version=GBu%2Felr%2Frefactor&_a=contents',
#     'repo_name': 'marlin',

#     # Visible levels of the global TOC; -1 means unlimited
#     'globaltoc_depth': 3,
#     # If False, expand all TOC entries
#     'globaltoc_collapse': False,
#     # If True, show hidden TOC entries
#     'globaltoc_includehidden': False,
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
