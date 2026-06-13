import xarray
import jax
from .custom_types import maybe_hash_coords, _HashableCoords
from typing import Tuple, Hashable, Mapping, Callable
import contextlib
import contextvars

VarChangeFn = Callable[[xarray.Variable], xarray.Variable]
_VAR_CHANGE_ON_UNFLATTEN_FN: contextvars.ContextVar[VarChangeFn] = (
    contextvars.ContextVar("var_change_on_unflatten_fn")
)


@contextlib.contextmanager
def var_change_on_unflatten(var_change_fn: VarChangeFn):
    """A context manager for modifying a variable on unflatten. This was inspired by the dims_change_on_unflatten function in the GraphCast project, but is ultimately a different approach."""
    token = _VAR_CHANGE_ON_UNFLATTEN_FN.set(var_change_fn)
    try:
        yield
    finally:
        _VAR_CHANGE_ON_UNFLATTEN_FN.reset(token)


def _flatten_variable(
    v: xarray.Variable,
) -> Tuple[Tuple[jax.typing.ArrayLike], Tuple[Hashable, ...]]:
    children = (v._data,)
    aux = (
        v._dims,
        v._attrs,
    )
    return children, aux


def _unflatten_variable(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.Variable:
    """Unflattens a Variable for jax.tree_util."""
    (
        dims,
        attrs,
    ) = aux
    (data,) = children

    var = object.__new__(xarray.Variable)
    var._dims = dims
    var._data = data
    var._attrs = attrs
    var._encoding = None

    var_change_fn = _VAR_CHANGE_ON_UNFLATTEN_FN.get(None)
    if var_change_fn is not None:
        var = var_change_fn(var)

    return var


def _flatten_data_array(
    da: xarray.DataArray,
):
    children = (da._variable,)
    aux = (da._name, maybe_hash_coords(da._coords), da._indexes)
    return children, aux


def _unflatten_data_array(
    aux: Tuple[Hashable, ...], children: Tuple[jax.typing.ArrayLike]
) -> xarray.DataArray:
    """Unflattens a DataArray for jax.tree_util."""
    name, coords, indexes = aux
    (variable,) = children
    da = object.__new__(xarray.DataArray)
    da._variable = variable
    da._name = name
    da._coords = dict(coords)
    da._indexes = indexes
    return da


def _flatten_dataset(
    ds: xarray.Dataset,
):
    coord_names = ds._coord_names
    variables = ds._variables

    coords = {name: variables[name] for name in coord_names}
    data_vars = {name: variables[name] for name in variables if name not in coord_names}

    children = (data_vars,)
    aux = (maybe_hash_coords(coords), ds._indexes, ds._dims, ds._attrs)
    assert isinstance(aux, Hashable)
    return children, aux


def _unflatten_dataset(
    aux: _HashableCoords,
    children: Tuple[
        Mapping[Hashable, xarray.Variable], Mapping[Hashable, xarray.Variable]
    ],
) -> xarray.Dataset:
    """Unflattens a Dataset for jax.tree_util."""
    data_vars = children[0]
    coords, indexes, dims, attrs = aux

    ds = object.__new__(xarray.Dataset)
    ds._dims = dims
    ds._variables = data_vars | dict(coords)
    ds._coord_names = set(coords.keys())
    ds._attrs = attrs
    ds._indexes = indexes
    ds._encoding = None
    ds._close = None
    return ds


jax.tree_util.register_pytree_node(
    xarray.Variable, _flatten_variable, _unflatten_variable
)
jax.tree_util.register_static(xarray.IndexVariable)
jax.tree_util.register_pytree_node(
    xarray.DataArray, _flatten_data_array, _unflatten_data_array
)
jax.tree_util.register_pytree_node(xarray.Dataset, _flatten_dataset, _unflatten_dataset)
