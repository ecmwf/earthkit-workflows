import numpy as np
import pytest

from earthkit.workflows.fluent import merge
from earthkit.workflows.nodetree import combine_by_coords, datacubes, nodetree_arrays

from .helpers import mock_action


def test_datacubes():
    action = merge(
        mock_action((2, 1), coords={"dim": [0, 1], "dim1": [0]}, path="/path1"),
        mock_action((2,), coords={"dim1": [1, 2]}, path="/path2"),
    )
    assert len(datacubes(action.nodes)) == 2


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
