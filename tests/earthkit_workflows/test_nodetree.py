from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from earthkit.workflows.nodetree import combine_by_coords, coords_to_list, datacubes, nodetree_arrays, nodetree_from_dict

from .helpers import mock_action


@pytest.mark.parametrize(
    "datatree",
    [
        {
            "/path1": xr.DataArray(np.empty((2, 1)), coords={"dim": [0, 1], "dim1": [0]}),
            "/path2": xr.DataArray(
                np.empty(
                    2,
                ),
                coords={"dim1": [1, 2]},
            ),
        },
        {
            "/path1": xr.DataArray(np.empty((2, 1)), coords={"dim": [datetime(2024, 1, 1), datetime(2024, 1, 2)], "dim1": [0]}),
            "/path2": xr.DataArray(
                np.empty(
                    2,
                ),
                coords={"dim": [datetime(2024, 1, 3), datetime(2024, 1, 4)]},
            ),
        },
    ],
    ids=["numeric", "datetime"],
)
def test_datacubes(datatree: dict):
    tree = nodetree_from_dict(datatree)
    cubes = datacubes(tree)
    assert len(cubes) == 2
    assert cubes[0]["dim"] == coords_to_list(datatree["/path1"].coords["dim"].data)


@pytest.mark.parametrize(
    "inputs, dims",
    [
        [[mock_action((1,), coords={"dim": [0]}).nodes, mock_action((1,), coords={"dim": [1]}).nodes], {"/": {"dim": [0, 1]}}],
        [
            [
                mock_action((1, 1), coords={"dim": [0], "dim1": [0]}).nodes,
                mock_action((1, 1), coords={"dim": [1], "dim1": [0]}).nodes,
                mock_action((1, 1), coords={"dim": [0], "dim1": [1]}).nodes,
                mock_action((1, 1), coords={"dim": [1], "dim1": [1]}).nodes,
            ],
            {"/": {"dim": [0, 1], "dim1": [0, 1]}},
        ],
        [
            [
                mock_action((1, 1), coords={"dim": [0], "dim1": [0]}, path="/path1").nodes,
                mock_action((1, 1), coords={"dim": [1], "dim1": [0]}, path="/path1").nodes,
                mock_action((1,), coords={"dim1": [1]}, path="/path2").nodes,
                mock_action((1,), coords={"dim1": [2]}, path="/path2").nodes,
            ],
            {"/path1": {"dim": [0, 1], "dim1": [0]}, "/path2": {"dim1": [1, 2]}},
        ],
    ],
)
def test_combine(inputs, dims):
    outputs = combine_by_coords(inputs)
    for npath, narray in nodetree_arrays(outputs):
        for dim, values in dims[npath].items():
            assert dim in narray.coords
            assert np.all(values == narray.coords[dim])
