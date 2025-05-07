# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools

import dill
import numpy as np
import pytest

from earthkit.workflows.fluent import Action, Payload, custom_hash, from_source
from earthkit.workflows.graph import deserialise, serialise

from .helpers import mock_action


def test_payload():
    payload = Payload(np.random.rand, (2, 3, 4))
    hash1 = custom_hash(f"{payload}")
    payload2 = Payload(np.random.rand, (2, 3, 4), {})
    hash2 = custom_hash(f"{payload2}")
    payload3 = Payload(np.random.rand, (2, 3, 4), {"test": 1})
    hash3 = custom_hash(f"{payload3}")
    assert hash1 == hash2
    assert payload == payload2
    assert hash1 != hash3
    assert payload != payload3


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
    assert action.nodes.shape == shape


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
    assert len(action2.nodes.data.item(0).inputs) == 2

    with pytest.raises(Exception):
        action2.flatten()

    action3 = action2.expand("dim_0", internal_dim=0, dim_size=2)
    assert action3.nodes.shape == (2,)
    assert len(action3.nodes.data.item(0).inputs) == 1

    action4 = action3.expand("dim_1", internal_dim=0, dim_size=3, axis=1)
    assert action4.nodes.shape == (2, 3)
    assert len(action4.nodes.data.item(0).inputs) == 1


@pytest.mark.parametrize(
    "input_nodes_shape, func, inputs, output_nodes_shape, node_inputs",
    [
        [(3, 4), "map", [Payload("test")], (3, 4), 1],  # type: ignore
        [(3, 4, 5), "reduce", [Payload("func")], (4, 5), 3],  # type: ignore
        [
            (3, 4, 5),
            "reduce",
            [Payload("func"), None, "dim_1"],  # type: ignore
            (3, 5),
            4,
        ],
        [(3,), "reduce", [Payload("func")], (), 3],  # type: ignore
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


@pytest.mark.skip("Serialisation not supported due to sinks with outputs")
def test_serialisation(tmpdir, task_graph):
    assert len(task_graph.sinks) > 0
    data = serialise(task_graph)
    with open(f"{tmpdir}/graph.dill", "wb") as f:
        dill.dump(data, f)

    with open(f"{tmpdir}/graph.dill", "rb") as f:
        read_data = dill.load(f)
    new_graph = deserialise(read_data)
    assert len(task_graph.sinks) == len(new_graph.sinks)


def test_invalid_registration():
    with pytest.raises(TypeError):
        Action.register("test", None)


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

    action = from_source(
        functools.partial(test_func, 10), ("val", list(range(0, 100, 10)))
    )
    assert action.nodes.shape == (10,)
    assert action.nodes.dims == ("val",)
    cas = action.map(
        functools.partial(test_func, length=5), ("map", list(range(5)))
    ).reduce(functools.partial(test_func, length=2), ("reduce", ["a", "b"]))
    assert cas.nodes.dims == ("map", "reduce")
    expected_coords = {"map": list(range(5)), "reduce": ["a", "b"]}
    for dim, vals in expected_coords.items():
        assert np.all(cas.nodes.coords[dim] == vals)
    assert cas.nodes.shape == (5, 2)
    graph = cas.graph()
    assert len(graph.sinks) == 5
    serialise(graph)
