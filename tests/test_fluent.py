import pytest
import xarray as xr
import numpy as np
import dill

from cascade.fluent import Payload, Node, SingleAction, MultiAction, Fluent
from cascade.graph import serialise, deserialise

from helpers import MockNode, backend, mock_action, mock_graph


@pytest.mark.parametrize(
    "func, args, kwargs, type",
    [
        [np.random.rand, (2, 3), {}, SingleAction],
        [
            np.random.rand,
            xr.DataArray([[{2, 3}, {2, 3}], [{2, 3}, {2, 3}]], dims=["x", "y"]),
            {},
            MultiAction,
        ],
        [
            np.random.rand,
            {2, 3},
            xr.DataArray(
                [[{"test": 2}, {"test": 3}], [{"test": 3}, {"test": 2}]],
                dims=["x", "y"],
            ),
            MultiAction,
        ],
        [
            xr.DataArray(
                [[np.random.rand, np.random.rand], [np.random.rand, np.random.rand]],
                dims=["x", "y"],
            ),
            xr.DataArray([[{2, 3}, {2, 3}], [{2, 3}, {2, 3}]], dims=["x", "y"]),
            {},
            MultiAction,
        ],
    ],
)
def test_source(func, args, kwargs, type):
    action = Fluent().source(func, args, kwargs)
    assert isinstance(action, type)
    if isinstance(action, MultiAction):
        assert action.nodes.shape == (2, 2)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        [
            xr.DataArray(
                [
                    [np.random.rand, np.random.rand],
                    [np.random.rand, np.random.rand],
                    [np.random.rand, np.random.rand],
                ],
                dims=["x", "y"],
            ),
            xr.DataArray([[{2, 3}, {2, 3}], [{2, 3}, {2, 3}]], dims=["x", "y"]),
            {},
        ],
        [
            xr.DataArray(
                [[np.random.rand, np.random.rand], [np.random.rand, np.random.rand]],
                dims=["x", "z"],
            ),
            xr.DataArray([[{2, 3}, {2, 3}], [{2, 3}, {2, 3}]], dims=["x", "y"]),
            {},
        ],
        [
            xr.DataArray(
                [[np.random.rand, np.random.rand], [np.random.rand, np.random.rand]],
                dims=["x", "y"],
                coords={"x": [1, 2], "y": [3, 4]},
            ),
            xr.DataArray([[{2, 3}, {2, 3}], [{2, 3}, {2, 3}]], dims=["x", "y"]),
            {},
        ],
    ],
)
def test_source_invalid(func, args, kwargs):
    with pytest.raises(ValueError):
        Fluent().source(func, args, kwargs)


@pytest.mark.parametrize(
    "previous, nodes",
    [
        [
            SingleAction.from_payload(None, Payload("test"), backend),
            xr.DataArray([MockNode("1")]),
        ],
        [None, xr.DataArray([MockNode("1"), MockNode("2")])],
    ],
)
def test_single_action(previous, nodes):
    if nodes.size > 1:
        with pytest.raises(Exception):
            SingleAction(previous, nodes, backend)
    else:
        SingleAction(previous, nodes, backend)


@pytest.mark.parametrize(
    "payload, previous",
    [
        [Payload("test"), SingleAction.from_payload(None, Payload("test"), backend)],
        [
            Payload("test"),
            mock_action((2, 2)),
        ],
    ],
)
def test_single_action_from_payload(payload, previous):
    single_action = SingleAction.from_payload(previous, payload, backend)
    assert single_action.nodes.size == 1
    assert single_action.nodes.shape == ()
    assert len(single_action.nodes.data[()].inputs) == previous.nodes.size


def test_broadcast():
    input_action = mock_action((2, 3))

    with pytest.raises(Exception):
        input_action.broadcast(mock_action((3, 3)))

    output_action = input_action.broadcast(mock_action((2, 3, 3)))
    assert type(output_action) == MultiAction
    assert output_action.nodes.shape == (2, 3, 3)
    assert len(output_action.nodes.data.item(0).inputs) == 1
    it = np.nditer(output_action.nodes, flags=["multi_index", "refs_ok"])
    for _ in it:
        print(it.multi_index)
        assert output_action.nodes[it.multi_index].item(0).inputs[
            "input0"
        ].parent == input_action.nodes[it.multi_index[:2]].item(0)


def test_flatten_expand():
    input_action = mock_action((2, 3))

    with pytest.raises(Exception):
        input_action.flatten(dim="dim_2")

    action1 = input_action.flatten(dim="dim_1")
    assert action1.nodes.shape == (2,)
    assert len(action1.nodes.data.item(0).inputs) == 3

    action2 = action1.flatten(dim="dim_0")
    assert type(action2) == SingleAction
    assert len(action2.nodes.data.item(0).inputs) == 2

    with pytest.raises(Exception):
        action2.flatten()

    action3 = action2.expand("dim_0", 2)
    assert action3.nodes.shape == (2,)
    assert len(action3.nodes.data.item(0).inputs) == 1

    action4 = action3.expand("dim_1", 3, new_axis=1)
    assert action4.nodes.shape == (2, 3)
    assert len(action4.nodes.data.item(0).inputs) == 1


@pytest.mark.parametrize(
    "input_nodes_shape, func, inputs, output_type, output_nodes_shape, node_inputs",
    [
        [(3, 4), "map", [Payload("test")], MultiAction, (3, 4), 1],
        [(3, 4, 5), "reduce", [Payload("func")], MultiAction, (4, 5), 3],
        [
            (3, 4, 5),
            "reduce",
            [Payload("func"), "dim_1"],
            MultiAction,
            (3, 5),
            4,
        ],
        [(3,), "reduce", [Payload("func")], SingleAction, (), 3],
        [
            (3,),
            "join",
            [
                mock_action((1,)),
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
                mock_action((3,)),
                "data_type",
            ],
            MultiAction,
            (2, 3),
            0,
        ],
        [
            (3,),
            "transform",
            [lambda action, x: action.expand("dim_1", x), [(4,), (4,), (4,)], "index"],
            MultiAction,
            (3, 4, 3),
            1,
        ],
        [(3, 4), "select", [{"dim_0": 1}], MultiAction, (4,), 0],
        [(3,), "select", [{"dim_0": 1}], SingleAction, (), 0],
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
    input_action = mock_action(input_nodes_shape)

    output_action = getattr(input_action, func)(*inputs)
    assert type(output_action) == output_type
    assert output_action.nodes.shape == output_nodes_shape
    assert len(output_action.nodes.data.item(0).inputs) == node_inputs


def test_join_fail():
    input_action = mock_action((3, 4))
    second_action = mock_action((3, 5))
    with pytest.raises(Exception):
        input_action.join(second_action, "new_dim")

    input_action.join(second_action, "dim_1")


def test_attributes():
    action = mock_action((3,))

    # Set attributes global to all nodes
    action.add_attributes({"expver": "0001"})
    assert action.nodes.attrs["expver"] == "0001"


def test_select_nodes():
    action = mock_action((3, 4))
    assert isinstance(action.node({"dim_0": 0}), np.ndarray)
    assert isinstance(action.node({"dim_0": 0, "dim_1": 1}), Node)


def test_serialisation(tmpdir):
    graph = mock_graph(backend, np.random.rand)
    assert len(graph.sinks) > 0
    data = serialise(graph)
    with open(f"{tmpdir}/graph.dill", "wb") as f:
        dill.dump(data, f)

    with open(f"{tmpdir}/graph.dill", "rb") as f:
        read_data = dill.load(f)
    new_graph = deserialise(read_data)
    assert len(graph.sinks) == len(new_graph.sinks)
