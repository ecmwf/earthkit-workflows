# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr
from qubed import Qube

NodetreeMappings = Union[dict[str, xr.DataArray], dict[str, xr.Dataset], dict[str, Union[xr.DataArray, xr.Dataset]]]


def nodetree_from_dict(data: NodetreeMappings, *args, **kwargs) -> xr.DataTree:
    new_data = {}
    name: str = ""
    for k, v in data.items():
        if isinstance(v, xr.DataArray):
            if v.size == 0:
                raise ValueError(f"Attempting to add empty node array at path {k}")
            if len(name) == 0:
                name = "nodeset"
            new_data[k] = v.to_dataset(name=name)
        elif isinstance(v, xr.Dataset):
            if len(v.data_vars) != 1:
                raise ValueError(f"NodeTree can only be created from dict with xr.Dataset with one variable, got {len(v.data_vars)}")
            if len(name) == 0:
                name = str(list(v.data_vars.keys())[0])
            new_data[k] = v
        else:
            raise ValueError("NodeTree can only be created from dict of xr.DataArray or xr.Dataset")
        if name != list(new_data[k].data_vars.keys())[0]:
            raise ValueError("NodeTree can only be created from dict of xr.DataArray or xr.Dataset with same variable name")
    tree = xr.DataTree.from_dict(new_data, *args, **kwargs)
    for leaf in tree.leaves:
        if np.any(leaf[name].isnull()):
            raise ValueError(f"Nodes in Action can not contain NaNs. Found NaN in nodeset {leaf.path}")
    if not tree.is_hollow:
        raise ValueError("Nodes in Action must be hollow datatree")
    return tree


def nodetree_arrays(nodetree: xr.DataTree) -> Iterable[Tuple[str, xr.DataArray]]:
    for leaf in nodetree.leaves:
        var: str = list(leaf.dataset.data_vars.keys())[0]  # type: ignore[assignment]
        yield leaf.path, leaf[var]


def nodetree_array(nodetree: xr.DataTree, path: Optional[str] = None) -> xr.DataArray:
    if path is None:
        if len(nodetree.leaves) != 1:
            raise ValueError("Multiple node arrays present, path must be specified to retrieve one")
        leaf = nodetree.leaves[0]
    else:
        if path not in nodetree.leaves:
            raise KeyError(f"Path {path} not found in nodetree")
        leaf = nodetree[path]
    var: str = list(leaf.dataset.data_vars.keys())[0]  # type: ignore[assignment]
    return leaf[var]  # type: ignore[return-value]


def nodetree_size(nodetree: xr.DataTree) -> int:
    size = 0
    for _, array in nodetree_arrays(nodetree):
        size += array.size
    return size


def nodetree_dimensions(nodetree: xr.DataTree) -> set[str]:
    return set.union(set(), *[arr.dims for _, arr in nodetree_arrays(nodetree)])  # type: ignore[invalid-argument-type]


def nodetree_new_dimension(nodetree: xr.DataTree, attempt: str = "tempindex") -> str:
    index = 0
    while attempt in nodetree_dimensions(nodetree):
        attempt = f"{attempt}{index}"
        index += 1
    return attempt


def coords_to_list(data: np.ndarray) -> list[Any]:
    if isinstance(data.dtype, np.datetime64):
        data = data.astype("datetime64[us]")
    out = data.tolist()
    if not isinstance(out, list):
        out = [out]
    return out


def datacubes(nodetree: xr.DataTree) -> list[dict]:
    qube = Qube.empty()
    for _, narray in nodetree_arrays(nodetree):
        dimensions = narray.coords.indexes.keys()
        indexes = {}
        base_datacube = {}
        for coord_name in narray.coords.keys():
            coord = narray.coords[coord_name]
            if len(coord.indexes) == 0:
                base_datacube[coord_name] = coords_to_list(coord.data)
            else:
                assert len(coord.indexes) == 1
                index = list(coord.indexes.keys())[0]
                if index != coord_name:
                    indexes.setdefault(index, []).append(coord_name)
        base_datacube.update({dim: coords_to_list(narray.coords[dim].data) for dim in dimensions if dim not in indexes})
        updates = []
        for index, dependants in indexes.items():
            index_update = []
            combined = [index] + dependants
            for combination in zip(*[coords_to_list(narray.coords[dep].data) for dep in combined]):
                index_update.append({combined[index]: combination[index] for index in range(len(combined))})
            updates.append(index_update)
        for unique_updates in itertools.product(*updates):
            qube = qube | Qube.from_datacube(dict(base_datacube, **{k: v for update in unique_updates for k, v in update.items()}))
    return list(qube.datacubes())


def combine_by_coords(nodetrees: list[xr.DataTree]) -> xr.DataTree:
    arrays: dict[str, list[xr.DataArray]] = {}
    for tree in nodetrees:
        for npath, narray in nodetree_arrays(tree):
            arrays.setdefault(npath, []).append(narray)
    combined: dict[str, Union[xr.DataArray, xr.Dataset]] = {}
    for npath, narrays in arrays.items():
        combined[npath] = xr.combine_by_coords(narrays, coords="different", compat="identical", join="exact")
    nodetree = nodetree_from_dict(combined)
    return nodetree
