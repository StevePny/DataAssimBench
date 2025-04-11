# -- Project information

project = 'DataAssimBench'
copyright = '2025, The DataAssimBench Authors'
author = 'The DataAssimBench Authors'

release = ''
version = ''

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'autoapi.extension'
]
autoapi_dirs = ['../dabench']
autoapi_options = ['members', 'undoc-members', 'show-inheritance',
                   'show-module-summary',  'special-members',
                   'imported-members']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
