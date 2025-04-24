# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Pronoms'
author = 'Michael Riffle'
from pronoms import __version__ as release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# Set the documentation logo
html_logo = '_static/logo.png'
html_theme = 'renku'
