import pytest
import xarray as xr
import numpy as np

from cascade.fluent import SingleAction, MultiAction

from helpers.mock import MockNode, MockPayload


@pytest.mark.parametrize(
    "payload, previous",
    [
        [MockPayload("test"), SingleAction(MockPayload("test"), None)],
        [
            MockPayload("test"),
            MultiAction(
                None,
                xr.DataArray(
                    [[MockNode("1"), MockNode("2")], [MockNode("3"), MockNode("4")]]
                ),
            ),
        ],
    ],
)
def test_single_action(payload, previous):
    single_action = SingleAction(payload, previous, None)
    assert single_action.nodes.size == 1
    assert single_action.nodes.shape == ()
    assert len(single_action.nodes.data[()].inputs) == previous.nodes.size


@pytest.mark.parametrize(
    "previous, nodes",
    [
        [SingleAction(MockPayload("test"), None), xr.DataArray([MockNode("1")])],
        [None, xr.DataArray([MockNode("1"), MockNode("2")])],
    ],
)
def test_single_action_from_node(previous, nodes):
    if nodes.size > 1:
        with pytest.raises(Exception):
            SingleAction(MockPayload("test"), previous, nodes)
    else:
        SingleAction(MockPayload("test"), previous, nodes)


@pytest.mark.parametrize(
    "input_nodes_shape, func, inputs, output_type, output_nodes_shape, node_inputs, num_sinks",
    [
        [(3, 4), "foreach", [MockPayload("test")], MultiAction, (3, 4), 1, 0],
        [(3, 4, 5), "reduce", [MockPayload("func")], MultiAction, (4, 5), 3, 0],
        [
            (3, 4, 5),
            "reduce",
            [MockPayload("func"), "dim_1"],
            MultiAction,
            (3, 5),
            4,
            0,
        ],
        [(1,), "reduce", [MockPayload("func")], SingleAction, (), 1, 0],
        [(3,), "reduce", [MockPayload("func")], SingleAction, (), 3, 0],
        [
            (3,),
            "join",
            [
                SingleAction(
                    MockPayload("test"),
                    None,
                    xr.DataArray(MockNode("1"), coords={"dim_0": [0]}, dims=["dim_0"]),
                ),
                "dim_0",
            ],
            MultiAction,
            (4,),
            0,
            0,
        ],
        [
            (3,),
            "join",
            [
                MultiAction(
                    None,
                    xr.DataArray(
                        [MockNode("1"), MockNode("2"), MockNode("3")],
                        coords={"dim_0": list(range(3))},
                        dims=["dim_0"],
                    ),
                ),
                "data_type",
            ],
            MultiAction,
            (2, 3),
            0,
            0,
        ],
        [(3, 4), "select", [{"dim_0": 1}], MultiAction, (4,), 0, 0],
        [(3,), "select", [{"dim_0": 1}], SingleAction, (), 0, 0],
    ],
)
def test_multi_action(
    input_nodes_shape,
    func,
    inputs,
    output_type,
    output_nodes_shape,
    node_inputs,
    num_sinks,
):
    nodes = np.empty(input_nodes_shape, dtype=object)
    nodes[:] = MockNode("1")
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
    assert len(output_action.sinks) == num_sinks


def test_attributes():
    action = MultiAction(
        None,
        xr.DataArray(
            [MockNode("1"), MockNode("2"), MockNode("3")],
            coords={"dim_0": list(range(3))},
            dims=["dim_0"],
        ),
    )

    # Set attributes global to all nodes
    action.add_attributes({"expver": "0001"})
    assert action.nodes.attrs["expver"] == "0001"
