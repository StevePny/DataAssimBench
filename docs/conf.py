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
    'autoapi.extension',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.napoleon'
]
autoapi_dirs = ['../dabench']
# Important: Because we're not including "undoc-members",
# you need to include a docstring on *everything* you want documented.
# Including in __init__.py for submodules.
autoapi_options = ['members', 'show-module-summary',
                   'special-members', 'imported-members']
                   
autodoc_typehints = 'description'
autoapi_member_order = 'groupwise'
autoapi_add_toctree_entry = True
autoapi_own_page_level = 'module'
napoleon_numpy_docstring = False
napoleon_google_docstring = True

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
