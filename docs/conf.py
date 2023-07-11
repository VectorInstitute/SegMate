# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', 'segmate')))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SegMate'
copyright = '2023, Vector Institute'
author = 'Vahid Reza Khazaie, Marshall Wang'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = [
    "numpy",
    "cv2",
    "torch",
    "segment_anything",
    "groundingdino"
]
autosummary_generate = True

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
