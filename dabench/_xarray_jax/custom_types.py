import collections
from collections.abc import Hashable, Iterator, Mapping
from typing import (
    Optional,
    Union,
)

import equinox as eqx
import jax
import xarray
import jax.numpy as jnp


def error_if_inside_jit():
    is_jit = isinstance(jnp.array(0), jax.core.Tracer)
    if is_jit:
        raise ValueError(
            "This function should not be called inside a jax.jit'ed function."
        )


def maybe_hash_coords(coords):
    if isinstance(coords, _HashableCoords):
        return coords
    return _HashableCoords(coords)


class _HashableCoords(collections.abc.Mapping):
    """
    Originally implemented in the GraphCast project:
    https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py

    Wraps a dict of xarray Variables as hashable, used for static coordinates.

    This needs to be hashable so that when an xarray.Dataset is passed to a
    jax.jit'ed function, jax can check whether it's seen an array with the
    same static coordinates(*) before or whether it needs to recompile the
    function for the new values of the static coordinates.

    (*) note jax_coords are not included in this; their value can be different
    on different calls without triggering a recompile.
    """

    def __init__(self, coord_vars: Mapping[Hashable, xarray.Variable]):
        self._variables = coord_vars

    def __repr__(self) -> str:
        return f"_HashableCoords({self._variables!r})"

    def __getitem__(self, key: Hashable) -> xarray.Variable:
        return self._variables[key]

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._variables)

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash(
                frozenset(
                    (name, var.data.tobytes()) for name, var in self._variables.items()
                )
            )
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, type(self)):
            return NotImplemented
        elif self._variables is other._variables:
            return True
        else:
            return self._variables.keys() == other._variables.keys() and all(
                variable.equals(other._variables[name])
                for name, variable in self._variables.items()
            )


class XjVariable(eqx.Module):
    data: jax.Array
    dims: tuple[str, ...] = eqx.field(static=True)
    attrs: Optional[Mapping] = eqx.field(static=True)

    def __init__(
        self, data: jax.Array, dims: tuple[str, ...], attrs: Optional[Mapping] = None
    ):
        self.data = data
        self.dims = dims
        self.attrs = attrs

    def to_xarray(self) -> xarray.Variable:
        if self.data is None:
            return None
        return xarray.Variable(dims=self.dims, data=self.data, attrs=self.attrs)

    @classmethod
    def from_xarray(cls, var: xarray.Variable) -> "XjVariable":
        return cls(data=var.data, dims=var.dims, attrs=var.attrs)


class XjDataArray(eqx.Module):
    variable: XjVariable
    coords: _HashableCoords = eqx.field(static=True)
    name: Optional[str] = eqx.field(static=True)

    def __init__(
        self,
        variable: XjVariable,
        coords: _HashableCoords,
        name: Optional[str] = None,
    ):
        self.variable = variable
        self.coords = coords
        self.name = name

    def to_xarray(self) -> xarray.DataArray:
        var = self.variable.to_xarray()
        if var is None:
            return None
        return xarray.DataArray(var, name=self.name, coords=dict(self.coords))

    @classmethod
    def from_xarray(cls, da: xarray.DataArray) -> "XjDataArray":
        return cls(
            XjVariable.from_xarray(da.variable), maybe_hash_coords(da.coords), da.name
        )


class XjDataset(eqx.Module):
    variables: dict[Hashable, XjVariable]
    coords: _HashableCoords = eqx.field(static=True)
    attrs: Optional[Mapping] = eqx.field(static=True)

    def __init__(
        self,
        variables: dict[Hashable, XjVariable],
        coords: _HashableCoords,
        attrs: Optional[Mapping] = None,
    ):
        self.variables = variables
        self.coords = coords
        self.attrs = attrs

    def to_xarray(self) -> xarray.Dataset:
        data_vars = {name: var.to_xarray() for name, var in self.variables.items()}

        data_vars = {name: var for name, var in data_vars.items() if var is not None}

        return xarray.Dataset(
            data_vars,
            coords=dict(self.coords),
            attrs=self.attrs,
        )

    @classmethod
    def from_xarray(cls, ds: xarray.Dataset) -> "XjDataset":
        return cls(
            {
                name: XjVariable.from_xarray(da.variable)
                for name, da in ds.data_vars.items()
            },
            maybe_hash_coords(ds.coords),
            ds.attrs,
        )


def from_xarray(
    obj: Union[xarray.Variable, xarray.DataArray, xarray.Dataset],
) -> Union[XjVariable, XjDataArray, XjDataset]:
    """Convert an xarray object to an xarray_jax object."""
    if isinstance(obj, xarray.Variable):
        return XjVariable.from_xarray(obj)
    elif isinstance(obj, xarray.DataArray):
        return XjDataArray.from_xarray(obj)
    elif isinstance(obj, xarray.Dataset):
        return XjDataset.from_xarray(obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def to_xarray(
    obj: Union[XjVariable, XjDataArray, XjDataset],
) -> Union[xarray.Variable, xarray.DataArray, xarray.Dataset]:
    """Convert an xarray_jax object to an xarray object."""
    if isinstance(obj, XjVariable):
        return obj.to_xarray()
    elif isinstance(obj, XjDataArray):
        return obj.to_xarray()
    elif isinstance(obj, XjDataset):
        return obj.to_xarray()
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")
