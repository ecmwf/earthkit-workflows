# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterable, Optional, Tuple

import numpy as np
import xarray as xr


def nodetree_from_dict(
    data: dict[str, xr.DataArray] | dict[str, xr.Dataset], *args, **kwargs
) -> xr.DataTree:
    new_data = {}
    for nindex, (k, v) in enumerate(data.items()):
        if isinstance(v, xr.Dataset):
            new_data[k] = v
        elif isinstance(v, xr.DataArray):
            new_data[k] = v.to_dataset(name=f"nodeset{nindex}")
        else:
            raise ValueError(
                "NodeTree can only be created from dict of xr.DataArray or xr.Dataset"
            )
    tree = xr.DataTree.from_dict(new_data, *args, **kwargs)
    for leaf in tree.leaves:
        var = list(leaf.dataset.data_vars.keys())[0]
        if np.any(leaf[var].isnull()):
            raise ValueError(
                "Nodes in Action can not contain NaNs. "
                f"Found NaN in nodeset {leaf.path}, variable {var}"
            )
    if not tree.is_hollow:
        raise ValueError("Nodes in Action must be hollow datatree")
    return tree


def nodetree_arrays(nodetree: xr.DataTree) -> Iterable[Tuple[str, xr.DataArray]]:
    for leaf in nodetree.leaves:
        var = list(leaf.dataset.data_vars.keys())[0]
        yield leaf.path, leaf[var]


def nodetree_array(nodetree: xr.DataTree, path: Optional[str] = None) -> xr.DataArray:
    if path is None:
        if len(nodetree.leaves) != 1:
            raise ValueError(
                "Multiple node arrays present, path must be specified to retrieve one"
            )
        leaf = nodetree.leaves[0]
    else:
        if path not in nodetree.leaves:
            raise KeyError(f"Path {path} not found in nodetree")
        leaf = nodetree[path]
    var = list(leaf.dataset.data_vars.keys())[0]
    return leaf[var]


def nodetree_size(nodetree: xr.DataTree) -> int:
    size = 0
    for _, array in nodetree_arrays(nodetree):
        size += array.size
    return size
