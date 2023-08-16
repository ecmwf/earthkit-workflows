import pytest
import xarray as xr
import numpy as np

from ppgraph import Node

from cascade.fluent import SingleAction, MultiAction


@pytest.mark.parametrize(
    "func, previous",
    [
        ["test", SingleAction("test", None)],
        [
            "test",
            MultiAction(
                None, xr.DataArray([[Node("1"), Node("2")], [Node("3"), Node("4")]])
            ),
        ],
    ],
)
def test_single_action_from_func(func, previous):
    single_action = SingleAction(func, previous, None)
    assert single_action.nodes.size == 1
    assert single_action.nodes.shape == (1,)
    assert len(single_action.nodes.data[0].inputs) == previous.nodes.size


@pytest.mark.parametrize(
    "previous, nodes",
    [
        [SingleAction("test", None), xr.DataArray([Node("1")])],
        [None, xr.DataArray([Node("1"), Node("2")])],
    ],
)
def test_single_action_from_node(previous, nodes):
    if nodes.size > 1:
        with pytest.raises(Exception):
            SingleAction(None, previous, nodes)
    else:
        SingleAction(None, previous, nodes)


@pytest.mark.parametrize(
    "input_nodes_shape, input_internal_dims, func, inputs, output_type, output_nodes_shape, output_internal_dims, node_inputs",
    [
        [(3, 4), 0, "map", ["test"], MultiAction, (3, 4), 0, 1],
        [(3, 4), 1, "groupby", ["dim_0"], MultiAction, (3,), 2, 4],
        [(3, 4, 5), 0, "groupby", ["dim_1"], MultiAction, (4,), 1, 15],
        [(3, 4, 5), 0, "reduce", ["func"], MultiAction, (4, 5), 0, 3],
        [(3, 4, 5), 0, "reduce", ["func", "dim_1"], MultiAction, (3, 5), 0, 4],
        [(1,), 2, "reduce", ["func"], MultiAction, (1,), 1, 1],
        [(1,), 1, "reduce", ["func"], SingleAction, (1,), 0, 1],
        [(3,), 0, "reduce", ["func"], SingleAction, (1,), 0, 3],
        [
            (3,),
            1,
            "join",
            [SingleAction("test", None), "dim_0"],
            MultiAction,
            (4,),
            1,
            0,
        ],
        [
            (3,),
            1,
            "join",
            [
                MultiAction(None, xr.DataArray([Node("1"), Node("2"), Node("3")])),
                "data_type",
            ],
            MultiAction,
            (2, 3),
            1,
            0,
        ],
        [(3, 4), 1, "select", ["dim_0", 1], MultiAction, (4,), 1, 0],
        [(3,), 1, "select", ["dim_0", 1], SingleAction, (1,), 0, 0],
        [(3, 4), 1, "write", [], MultiAction, (3, 4), 1, 1],
    ],
)
def test_multi_action(
    input_nodes_shape,
    input_internal_dims,
    func,
    inputs,
    output_type,
    output_nodes_shape,
    output_internal_dims,
    node_inputs,
):
    nodes = np.empty(input_nodes_shape, dtype=object)
    nodes[:] = Node("1")
    input_action = MultiAction(None, xr.DataArray(nodes), input_internal_dims)

    output_action = getattr(input_action, func)(*inputs)
    assert type(output_action) == output_type
    assert output_action.nodes.shape == output_nodes_shape
    if output_type == MultiAction:
        assert output_action.internal_dims == output_internal_dims
    assert len(output_action.nodes.data.item(0).inputs) == node_inputs
