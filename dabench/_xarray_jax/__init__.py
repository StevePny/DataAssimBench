"""Vendored xarray + JAX integration.

Originally based on https://github.com/kysolvik/xarray_jax_permissible
(MIT licensed), itself derived from the GraphCast xarray_jax module.
Absorbed into DataAssimBench to avoid an external git dependency and the
associated equinox version pin in upstream.
"""
import xarray

from .custom_types import (
    XjDataArray,
    XjDataset,
    XjVariable,
    from_xarray,
    to_xarray,
)
from .register_pytrees import var_change_on_unflatten

xarray.set_options(keep_attrs=True)  # Necessary for preserving PyTree structure.

__all__ = [
    "XjDataArray",
    "XjDataset",
    "XjVariable",
    "from_xarray",
    "to_xarray",
    "var_change_on_unflatten",
]
