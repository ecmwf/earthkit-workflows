# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
from datetime import datetime

import numpy as np
import pytest

from earthkit.workflows.fluent import Action, from_source, merge
from earthkit.workflows.graph import serialise
from earthkit.workflows.nodetree import (
    nodetree_array,
    nodetree_arrays,
)

from .helpers import mock_action


@pytest.mark.parametrize(
    "payloads, dims, coords, shape",
    [
        [functools.partial(np.random.rand, 2, 3), None, None, ()],
        [
            [
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
            ],
            ["x", "y"],
            {"x": [0, 1], "y": [1, 2]},
            (2, 2),
        ],
    ],
)
def test_source(payloads, dims, coords, shape):
    action = from_source(payloads, dims=dims, coords=coords)
    narrays = list(nodetree_arrays(action.nodes))
    assert len(narrays) == 1
    assert narrays[0][1].shape == shape


@pytest.mark.parametrize(
    "payloads, dims, coords",
    [
        [
            [
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
            ],
            ["x"],
            None,
        ],
        [
            [
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
                [
                    functools.partial(np.random.rand, 2, 3),
                    functools.partial(np.random.rand, 2, 3),
                ],
            ],
            None,
            {"x": [1, 2], "y": [3, 4]},
        ],
    ],
    ids=["invalid_dims", "invalid_coords"],
)
def test_source_invalid(payloads, dims, coords):
    with pytest.raises(ValueError):
        from_source(payloads, dims=dims, coords=coords)


def test_broadcast():
    input_action = mock_action((2, 3))

    with pytest.raises(Exception):
        input_action.broadcast(mock_action((3, 3)))

    output_action = input_action.broadcast(mock_action((2, 3, 3)))
    out_array = nodetree_array(output_action.nodes)
    assert out_array.shape == (2, 3, 3)
    assert len(out_array.data.item(0).inputs) == 1
    it = np.nditer(out_array, flags=["multi_index", "refs_ok"])  # type: ignore[call-overload]
    for _ in it:
        print(it.multi_index)
        assert out_array[it.multi_index].item(0).inputs["1"].parent == nodetree_array(input_action.nodes)[it.multi_index[:2]].item(0)


def test_flatten_expand():
    input_action = mock_action((2, 3))

    with pytest.raises(ValueError):
        input_action.flatten(new_dim="temp", keep_dims=["dim_2"])
    action1 = input_action.flatten(new_dim="temp", keep_dims=["dim_0"]).concatenate(dim="temp")
    action1_array = nodetree_array(action1.nodes)
    assert action1_array.shape == (2,)
    assert len(action1_array.data.item(0).inputs) == 3

    action2 = action1.flatten(new_dim="temp").stack(dim="temp")
    assert len(nodetree_array(action2.nodes).data.item(0).inputs) == 2

    flatten_all = input_action.flatten(new_dim="temp").concatenate(dim="temp")
    assert flatten_all.nodes == action2.nodes

    action3 = action2.expand("dim_0", internal_dim=0, dim_size=2)
    action3_array = nodetree_array(action3.nodes)
    assert action3_array.shape == (2,)
    assert len(action3_array.data.item(0).inputs) == 1

    action4 = action3.expand("dim_1", internal_dim=0, dim_size=3, axis=1)
    action4_array = nodetree_array(action4.nodes)
    assert action4_array.shape == (2, 3)
    assert len(action4_array.data.item(0).inputs) == 1


@pytest.mark.parametrize(
    "input_nodes_shape, func, inputs, output_nodes_shape, node_inputs",
    [
        [(3, 4), "map", ["test"], (3, 4), 1],  # type: ignore
        [(3, 4, 5), "reduce", ["func"], (4, 5), 3],  # type: ignore
        [
            (3, 4, 5),
            "reduce",
            ["func", None, "dim_1"],  # type: ignore
            (3, 5),
            4,
        ],
        [(3,), "reduce", ["func"], (), 3],  # type: ignore
        [
            (3,),
            "join",
            [
                mock_action((1,)),
                "dim_0",
            ],
            (4,),
            0,
        ],
        [
            (3,),
            "join",
            [
                mock_action((3,)),
                "data_type",
            ],
            (2, 3),
            0,
        ],
        [
            (3,),
            "transform",
            [
                lambda action, x: action.expand("dim_1", internal_dim=0, dim_size=x),
                [(4,), (4,), (4,)],
                "index",
            ],
            (3, 4, 3),
            1,
        ],
        [(3, 4), "select", [{"dim_0": 1}], (4,), 0],
        [(3,), "select", [{"dim_0": 1}], (), 0],
    ],
)
def test_multi_action(
    input_nodes_shape,
    func,
    inputs,
    output_nodes_shape,
    node_inputs,
):
    input_action = mock_action(input_nodes_shape)

    output_action = getattr(input_action, func)(*inputs)
    assert nodetree_array(output_action.nodes).shape == output_nodes_shape
    assert len(nodetree_array(output_action.nodes).data.item(0).inputs) == node_inputs


def test_join_fail():
    input_action = mock_action((3, 4))
    second_action = mock_action((3, 5))
    with pytest.raises(Exception):
        input_action.join(second_action, "new_dim")

    input_action.join(second_action, "dim_1")


def test_invalid_registration():
    with pytest.raises(TypeError):
        Action.register("test", None)  # type: ignore[arg-type]


def test_registration():
    action = from_source(lambda x: x)

    class TestingAction(Action):
        def test_function(self):
            return self

    Action.register("test", TestingAction)
    assert hasattr(action, "test")
    assert hasattr(action.test, "test_function")


def test_dual_registration():
    Action.flush_registry()

    class TestingAction(Action):
        def test_function(self):
            return self

    Action.register("test", TestingAction)
    with pytest.raises(ValueError):
        Action.register("test", TestingAction)


def test_generators():
    def test_func(length: int, *multipliers):
        for val in range(length):
            yield val * sum([1, *multipliers])

    action = from_source(functools.partial(test_func, 10), ("val", list(range(0, 100, 10))))
    narray = nodetree_array(action.nodes)
    assert narray.shape == (10,)
    assert narray.dims == ("val",)
    cas = action.map(functools.partial(test_func, length=5), ("map", list(range(5)))).reduce(
        functools.partial(test_func, length=2), ("reduce", ["a", "b"])
    )
    new_narray = nodetree_array(cas.nodes)
    assert new_narray.dims == ("map", "reduce")
    expected_coords = {"map": list(range(5)), "reduce": ["a", "b"]}
    for dim, vals in expected_coords.items():
        assert np.all(new_narray.coords[dim] == vals)
    assert new_narray.shape == (5, 2)
    graph = cas.graph()
    assert len(graph.sinks) == 5
    serialise(graph)


@pytest.mark.parametrize(
    "branch_config",
    [
        {
            "/branch1": lambda data: np.where(data <= 0, data, np.nan),
            "/branch2": lambda data: np.where(data > 0, data, np.nan),
        },
        {
            "/branch1/subbranch1": lambda data: np.where(data < 0, data, np.nan),
            "/branch1/subbranch2": lambda data: np.where(data == 0, data, np.nan),
            "/branch2": lambda data: np.where(data == 0, data, np.nan),
        },
    ],
    ids=["branches", "subranches"],
)
@pytest.mark.parametrize(
    "combine_dim",
    [
        "dim_0",
        "dim_new",
    ],
    ids=["existing-dim", "new-dim"],
)
def test_branches(branch_config, combine_dim):
    input_action = mock_action((3, 4))
    branches = input_action.create_branches(branch_config)
    assert set(x[0] for x in nodetree_arrays(branches.nodes)) == set(branch_config.keys())
    for npath, narray in nodetree_arrays(branches.nodes):
        assert narray.shape == (3, 4)
        assert "branch" in npath
    recombined = branches.combine_branches(dim=combine_dim)
    for path, array in nodetree_arrays(recombined.nodes):
        assert path == "/"
        if combine_dim == "dim_new":
            assert array.shape == ((len(branch_config)), 3, 4)
        else:
            assert array.shape == ((len(branch_config)) * 3, 4)


def test_invalid_branches():
    input_action = mock_action((3, 4))
    branches = input_action.create_branches(
        {
            "/branch1": lambda data: np.where(data <= 0, data, np.nan),
            "/branch2": lambda data: np.where(data > 0, data, np.nan),
        }
    )
    with pytest.raises(NotImplementedError):
        branches.set_path("/new_root")

    with_root = input_action.set_path("/root")
    with pytest.raises(ValueError):
        with_root.create_branches(
            {
                "/branch1": lambda data: np.where(data <= 0, data, np.nan),
                "/branch2": lambda data: np.where(data > 0, data, np.nan),
            }
        )


def test_combine_branches():
    branches = merge(
        mock_action((3, 4))
        .set_path("/branch1")
        .create_branches(
            {
                "/branch1/subbranch1": lambda data: np.where(data < 0, data, np.nan),
                "/branch1/subbranch2": lambda data: np.where(data == 0, data, np.nan),
            },
        ),
        mock_action((5, 4, 6)).set_path("/branch2"),
    )
    assert len([x for x in nodetree_arrays(branches.combine_branches(dim="dim_0", path="/branch1").nodes)]) == 2
    reduced = branches.sum(path="/branch1/subbranch1")
    reduced.nodes["/branch2"].coords["scalar_dim"] = 1
    reduced.nodes["/branch1/subbranch1"].coords["scalar_dim"] = 2
    reduced.nodes["/branch1/subbranch2"].coords["scalar_dim"] = 2
    with pytest.raises(Exception, match="cannot align objects with join='exact"):
        reduced.combine_branches("dim_1")
    force = reduced.combine_branches(dim="dim_1", force=True)
    for _, array in nodetree_arrays(force.nodes):
        assert array.shape == (12,)


def test_flatten_branches():
    input_action = mock_action((3, 4))
    branches = input_action.create_branches(
        {
            "/branch1/subbranch1": lambda data: np.where(data < 0, data, np.nan),
            "/branch1/subbranch2": lambda data: np.where(data == 0, data, np.nan),
            "/branch2": lambda data: np.where(data == 0, data, np.nan),
        }
    )
    reduced = branches.flatten(new_dim="temp", path="/branch1/subbranch1").concatenate(dim="temp")
    flattened = reduced.flatten(new_dim="temp").concatenate(dim="temp")
    assert reduced.sel(path="/branch1/subbranch1").nodes == flattened.sel(path="/branch1/subbranch1").nodes


@pytest.mark.parametrize(
    "selection, num_arrays, shapes_or_error",
    [
        ({"dim_0": 1}, 3, [(5,), (4,), (4,)]),
        ({"dim_1": 4}, 1, [(2,)]),
        ({"path": "/branch1"}, 2, [(3, 4), (3, 4)]),
        ({"path": "/branch1", "dim_0": 1}, 2, [(4,), (4,)]),
        ({"date": [datetime(2024, 1, 1)]}, 1, [(2, 5)]),
        ({"dim_1": 10}, 0, IndexError),
        ({"dim_0": [2], "dim_1": [0, 4]}, 0, IndexError),
        ({"dim_0": [2], "dim_1": [0, 4], "expand": True}, 2, [(1, 1), (1, 1)]),
    ],
    ids=["in-all", "in-one", "by-path", "by-path-and-dim", "by-coord", "nonexistent", "no-expand", "expand"],
)
def test_select(selection, num_arrays, shapes_or_error):
    branches = merge(
        branch1=mock_action((3, 4)),
        branch2=mock_action((2, 5)),
    )
    subbranches = branches.create_branches(
        {
            "/branch1/subbranch1": lambda data: np.where(data < 0, data, np.nan),
            "/branch1/subbranch2": lambda data: np.where(data == 0, data, np.nan),
        }
    )
    subbranches.nodes["/branch2"].coords["date"] = datetime(2024, 1, 1)
    if num_arrays > 0:
        select_dim = subbranches.sel(**selection)
        assert len(list(nodetree_arrays(select_dim.nodes))) == num_arrays
        for index, (_, narray) in enumerate(nodetree_arrays(select_dim.nodes)):
            assert narray.shape == shapes_or_error[index]
    else:
        with pytest.raises(shapes_or_error):
            subbranches.sel(**selection)


@pytest.mark.parametrize(
    "selection, num_arrays, shapes_or_error",
    [
        ({"dim_0": 1}, 3, [(5,), (4,), (4,)]),
        ({"dim_1": 4}, 1, [(3,)]),
        ({"path": "/branch1"}, 2, [(3, 4), (3, 4)]),
        ({"path": "/branch1", "dim_0": 1}, 2, [(4,), (4,)]),
        ({"dim_1": 10}, 0, IndexError),
    ],
    ids=["in-all", "in-one", "by-path", "by-path-and-dim", "nonexistent"],
)
def test_iselect(selection, num_arrays, shapes_or_error):
    branches = merge(
        branch1=mock_action((3, 4)),
        branch2=mock_action((3, 5)),
    )
    subbranches = branches.create_branches(
        {
            "/branch1/subbranch1": lambda data: np.where(data < 0, data, np.nan),
            "/branch1/subbranch2": lambda data: np.where(data == 0, data, np.nan),
        }
    )
    if num_arrays > 0:
        select_dim = subbranches.isel(**selection)
        assert len(list(nodetree_arrays(select_dim.nodes))) == num_arrays
        for index, (_, narray) in enumerate(nodetree_arrays(select_dim.nodes)):
            assert narray.shape == shapes_or_error[index]
    else:
        with pytest.raises(shapes_or_error):
            subbranches.isel(**selection)


@pytest.mark.parametrize(
    "args, kwargs, dims",
    [
        [
            [],
            {"branch1": mock_action((3, 4)), "branch2": mock_action((3, 4))},
            {"/branch1": {"dim_0": [0, 1, 2], "dim_1": [0, 1, 2, 3]}, "/branch2": {"dim_0": [0, 1, 2], "dim_1": [0, 1, 2, 3]}},
        ],
        [
            [mock_action((3, 4)).set_path("/branch1"), mock_action((3, 4)).set_path("/branch2")],
            {},
            {"/branch1": {"dim_0": [0, 1, 2], "dim_1": [0, 1, 2, 3]}, "/branch2": {"dim_0": [0, 1, 2], "dim_1": [0, 1, 2, 3]}},
        ],
        [[mock_action((1,), coords={"dim": [0]}), mock_action((1,), coords={"dim": [1]})], {}, {"/": {"dim": [0, 1]}}],
        [
            [
                mock_action((1, 1), coords={"dim": [0], "dim1": [0]}),
                mock_action((1, 1), coords={"dim": [1], "dim1": [0]}),
                mock_action((1, 1), coords={"dim": [0], "dim1": [1]}),
                mock_action((1, 1), coords={"dim": [1], "dim1": [1]}),
            ],
            {},
            {"/": {"dim": [0, 1], "dim1": [0, 1]}},
        ],
        [
            [
                mock_action((1, 1), coords={"dim": [0], "dim1": [0]}, path="/path1"),
                mock_action((1, 1), coords={"dim": [1], "dim1": [0]}, path="/path1"),
                mock_action((1,), coords={"dim1": [1]}, path="/path2"),
                mock_action((1,), coords={"dim1": [2]}, path="/path2"),
            ],
            {},
            {"/path1": {"dim": [0, 1], "dim1": [0]}, "/path2": {"dim1": [1, 2]}},
        ],
    ],
    ids=["branches-with-args", "branches-with-kwargs", "single-coord", "multi-coords", "branches-and-coords"],
)
def test_merge(args, kwargs, dims):
    output = merge(*args, **kwargs)
    for npath, narray in nodetree_arrays(output.nodes):
        for dim, values in dims[npath].items():
            assert dim in narray.coords
            assert np.all(values == narray.coords[dim])


@pytest.mark.parametrize(
    "args, shape_or_error, coords",
    [
        [[{"new_dim": "x"}], (1, 4), {"dim_0": [0], "dim_1": [0, 1, 2, 3], "new_dim": "x"}],
        [[{"new_dim": "x"}, False, True], (1, 1, 4), {"dim_0": [0], "dim_1": [0, 1, 2, 3], "new_dim": ["x"]}],
        [[{"dim_0": 2}], ValueError, {}],
        [[{"dim_0": 2}, True, True], (1, 4), {"dim_0": [2], "dim_1": [0, 1, 2, 3]}],
    ],
    ids=["new-coord", "new-coord-expand", "existing-coord", "override-existing-coord"],
)
def test_set_coords(args, shape_or_error, coords):
    action = mock_action((1, 4))
    if isinstance(shape_or_error, type) and issubclass(shape_or_error, Exception):
        with pytest.raises(shape_or_error):
            action.set_scalar_coords(*args)
    else:
        action.set_scalar_coords(*args)
        for _, narray in nodetree_arrays(action.nodes):
            assert narray.shape == shape_or_error
            assert {dim: val.data.tolist() for dim, val in narray.coords.items()} == coords
