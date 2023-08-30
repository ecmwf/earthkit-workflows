import pytest
import xarray as xr
import numpy as np

from cascade.fluent import Node, SingleAction, MultiAction


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
    assert single_action.nodes.shape == ()
    assert len(single_action.nodes.data[()].inputs) == previous.nodes.size


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
    "input_nodes_shape, func, inputs, output_type, output_nodes_shape, node_inputs",
    [
        [(3, 4), "foreach", ["test"], MultiAction, (3, 4), 1],
        [(3, 4, 5), "reduce", ["func"], MultiAction, (4, 5), 3],
        [(3, 4, 5), "reduce", ["func", "dim_1"], MultiAction, (3, 5), 4],
        [(1,), "reduce", ["func"], SingleAction, (), 1],
        [(3,), "reduce", ["func"], SingleAction, (), 3],
        [
            (3,),
            "join",
            [
                SingleAction(
                    "test",
                    None,
                    xr.DataArray(Node("1"), coords={"dim_0": [0]}, dims=["dim_0"]),
                ),
                "dim_0",
            ],
            MultiAction,
            (4,),
            0,
        ],
        [
            (3,),
            "join",
            [
                MultiAction(
                    None,
                    xr.DataArray(
                        [Node("1"), Node("2"), Node("3")],
                        coords={"dim_0": list(range(3))},
                        dims=["dim_0"],
                    ),
                ),
                "data_type",
            ],
            MultiAction,
            (2, 3),
            0,
        ],
        [(3, 4), "select", ["dim_0", 1], MultiAction, (4,), 0],
        [(3,), "select", ["dim_0", 1], SingleAction, (), 0],
        [(3, 4), "write", [], MultiAction, (12,), 1],
    ],
)
def test_multi_action(
    input_nodes_shape,
    func,
    inputs,
    output_type,
    output_nodes_shape,
    node_inputs,
):
    nodes = np.empty(input_nodes_shape, dtype=object)
    nodes[:] = Node("1")
    input_action = MultiAction(
        None,
        xr.DataArray(
            nodes,
            coords={
                f"dim_{index}": list(range(shape))
                for index, shape in enumerate(input_nodes_shape)
            },
            dims=[f"dim_{index}" for index in range(len(input_nodes_shape))],
        ),
    )

    output_action = getattr(input_action, func)(*inputs)
    assert type(output_action) == output_type
    assert output_action.nodes.shape == output_nodes_shape
    assert len(output_action.nodes.data.item(0).inputs) == node_inputs
