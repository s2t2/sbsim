import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Smart Control'
copyright = '2024, Google'
author = 'Google'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'autoapi.extension',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

autoapi_type = 'python'
autoapi_dirs = ['../smart_control']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]
autoapi_ignore = ['*migrations*', '*test*']
