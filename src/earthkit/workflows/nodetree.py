# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr
from qubed import Qube

NodetreeMappings = Union[dict[str, xr.DataArray], dict[str, xr.Dataset], dict[str, Union[xr.DataArray, xr.Dataset]]]


def nodetree_from_dict(data: NodetreeMappings, *args, **kwargs) -> xr.DataTree:
    new_data = {}
    for nindex, (k, v) in enumerate(data.items()):
        if isinstance(v, xr.Dataset):
            new_data[k] = v
        elif isinstance(v, xr.DataArray):
            new_data[k] = v.to_dataset(name=f"nodeset{nindex}")
        else:
            raise ValueError("NodeTree can only be created from dict of xr.DataArray or xr.Dataset")
    tree = xr.DataTree.from_dict(new_data, *args, **kwargs)
    for leaf in tree.leaves:
        var: str = list(leaf.dataset.data_vars.keys())[0]  # type: ignore[assignment]
        if np.any(leaf[var].isnull()):
            raise ValueError(f"Nodes in Action can not contain NaNs. Found NaN in nodeset {leaf.path}, variable {var}")
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


def datacubes(nodetree: xr.DataTree) -> list[dict]:
    qube = Qube.empty()
    for _, narray in nodetree_arrays(nodetree):
        dimensions = narray.coords.indexes.keys()
        indexes = {}
        base_datacube = {}
        for coord_name in narray.coords.keys():
            coord = narray.coords[coord_name]
            if len(coord.indexes) == 0:
                base_datacube[coord_name] = coord.data.tolist()
            else:
                assert len(coord.indexes) == 1
                index = list(coord.indexes.keys())[0]
                if index != coord_name:
                    indexes.setdefault(index, []).append(coord_name)
        base_datacube.update({dim: narray.coords[dim].data.tolist() for dim in dimensions if dim not in indexes})
        updates = []
        for index, dependants in indexes.items():
            index_update = []
            combined = [index] + dependants
            for combination in zip(*[narray.coords[dep].data.tolist() for dep in combined]):
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
        combined[npath] = xr.combine_by_coords(narrays, coords="different", compat="identical")
    nodetree = nodetree_from_dict(combined)
    return nodetree
