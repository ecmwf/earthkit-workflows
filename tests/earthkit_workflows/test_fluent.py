# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import pytest
from qubed import Qube  # type: ignore

from earthkit.workflows.fluent import Action, Payload, from_source, merge
from earthkit.workflows.graph import deduplicate_nodes, serialise
from earthkit.workflows.nodeqube import Datacube, NodeKey


def action_from_shape(shape: Tuple[int, ...]) -> Action:
    datacubes = {f"dim_{i}": list(range(dim)) for i, dim in enumerate(shape)}
    return from_source("test", datacubes=datacubes)


class TestFromSource:
    @pytest.mark.parametrize(
        "payloads, datacubes, num_nodes",
        [
            [{NodeKey({"x": 1}): functools.partial(np.random.rand, 2, 3)}, None, 1],
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
                [{"x": [0, 1], "y": [1, 2]}],
                4,
            ],
        ],
    )
    def test_action_creation(self, payloads, datacubes, num_nodes):
        action = from_source(payloads, datacubes=datacubes)
        assert len(action.nodes) == num_nodes

    @pytest.mark.parametrize(
        "payloads, datacubes, match",
        [
            [
                "func",
                None,
                "If datacubes is None, payloads must be a dict of payloads",
            ],
            [
                {
                    NodeKey({"x": 0, "y": 1}): "func1",
                    NodeKey({"x": 1, "y": 1}): "func2",
                    NodeKey({"x": 0, "y": 2}): "func3",
                },
                [{"x": [0, 1], "y": [1, 2]}],
                "Length of payloads dict must match length of unique datacubes",
            ],
            [
                {NodeKey({"x": 2}): "func"},
                [{"x": 1}],
                "Missing payload for datacube",
            ],
        ],
        ids=[
            "payloads-not-dict",
            "payloads-length-mismatch",
            "missing-payload-for-datacube",
        ],
    )
    def test_invalid(self, payloads: Payload | dict[NodeKey, Payload], datacubes: list[Datacube] | None, match: str):
        with pytest.raises(ValueError, match=match):
            from_source(payloads, datacubes=datacubes)


class TestRegistration:
    def test_invalid_registration(self):
        with pytest.raises(TypeError):
            Action.register("test", None)  # type: ignore[arg-type]

    def test_registration(self):
        action = from_source(lambda x: x, datacubes=[{"dim_0": [0]}])

        class TestingAction(Action):
            def test_function(self):
                return self

        Action.register("test", TestingAction)
        assert hasattr(action, "test")
        assert hasattr(action.test, "test_function")

    def test_dual_registration(self):
        Action.flush_registry()

        class TestingAction(Action):
            def test_function(self):
                return self

        Action.register("test", TestingAction)
        with pytest.raises(ValueError):
            Action.register("test", TestingAction)


class TestFluentMethods:
    def test_broadcast(self):
        input_action = action_from_shape((2, 3))
        assert len(input_action.nodeqube) == 6

        with pytest.raises(Exception):
            input_action.broadcast(action_from_shape((3, 3)))

        output_action = input_action.broadcast(action_from_shape((2, 3, 3)))

        assert len(output_action.nodeqube) == 18
        for key, node in output_action.nodes.items():
            assert len(node.inputs) == 1
            cube = key.to_datacube()
            cube.pop("dim_2")
            inputs = input_action.select(cube)
            assert len(inputs.nodes) == 1
            assert node.inputs["1"].parent == list(inputs.nodes.values())[0]

    def test_flatten_expand(self):
        input_action = action_from_shape((2, 3))

        # Non-existent dimension in keep_dims should raise ValueError
        with pytest.raises(ValueError, match="Keep dimensions contain dimensions not in qube"):
            input_action.flatten(new_dim="temp", keep_dims=["dim_2"])

        action1 = input_action.flatten(new_dim="temp", keep_dims=["dim_0"]).concatenate(dim="temp")
        assert len(action1.nodes) == 2
        for node in action1.nodes.values():
            assert len(node.inputs) == 3

        action2 = action1.flatten(new_dim="temp").stack(dim="temp")
        for node in action2.nodes.values():
            assert len(node.inputs) == 2

        flatten_all = input_action.flatten(new_dim="temp").concatenate(dim="temp")
        for node in flatten_all.nodes.values():
            assert len(node.inputs) == 6

        action3 = action2.expand("dim_0", internal_dim=0, dim_size=2)
        assert len(action3.nodes) == 2
        for node in action3.nodes.values():
            assert len(node.inputs) == 1

        action4 = action3.expand("dim_1", internal_dim=0, dim_size=3)
        assert len(action4.nodes) == 6
        for node in action4.nodes.values():
            assert len(node.inputs) == 1

    @pytest.mark.parametrize(
        "input_nodes_shape, func, inputs, output_nodes_shape, node_inputs",
        [
            [(3, 4), "map", ["test"], {"dim_0": 3, "dim_1": 4}, 1],  # type: ignore
            [(3, 4, 5), "reduce", ["func", "dim_0"], {"dim_1": 4, "dim_2": 5}, 3],  # type: ignore
            [
                (3, 4, 5),
                "reduce",
                ["func", "dim_1"],  # type: ignore
                {"dim_0": 3, "dim_2": 5},
                4,
            ],
            [(3,), "reduce", ["func", "dim_0"], {"dim_0": 1}, 3],  # type: ignore
            [
                (3,),
                "join",
                [
                    from_source("test", datacubes=[{"dim_0": 3}]),
                ],
                {"dim_0": 4},
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
                {"dim_0": 3, "dim_1": 4, "index": 3},
                1,
            ],
            [(3, 4), "select", [{"dim_0": 1}], {"dim_0": 1, "dim_1": 4}, 0],
            [(3,), "select", [{"dim_0": 1}], {"dim_0": 1}, 0],
        ],
    )
    def test_multi_action(
        self,
        input_nodes_shape: Tuple[int, ...],
        func: str,
        inputs: list[Any],
        output_nodes_shape: dict[str, int],
        node_inputs: int,
    ):
        input_action = action_from_shape(input_nodes_shape)

        output_action = getattr(input_action, func)(*inputs)
        datacubes = list(output_action.nodeqube.datacubes())
        assert len(datacubes) == 1
        assert tuple(
            1 if not isinstance(datacubes[0][dim], list) else len(datacubes[0][dim]) for dim in output_nodes_shape.keys()
        ) == tuple(output_nodes_shape.values())
        for node in output_action.nodes.values():
            assert len(node.inputs) == node_inputs

    def test_join_fail(self):
        input_action = action_from_shape((3, 4))
        second_action = action_from_shape((3, 5))
        with pytest.raises(ValueError, match="Node key conflict for key"):
            input_action.join(second_action)

    def test_generators(self):
        def test_func(length: int, *multipliers):
            for val in range(length):
                yield val * sum([1, *multipliers])

        action = from_source({NodeKey({"dim_0": 0}): functools.partial(test_func, 10)}, yields=("val", list(range(0, 100, 10))))
        datacubes = list(action.nodeqube.datacubes())
        assert len(datacubes) == 1
        assert datacubes[0] == {"dim_0": 0, "val": list(range(0, 100, 10))}
        cas = action.map(functools.partial(test_func, length=5), yields=("map", list(range(5)))).reduce(
            functools.partial(test_func, length=2), dim="val", yields=("reduce", ["a", "b"])
        )
        new_datacubes = list(cas.nodeqube.datacubes())
        assert new_datacubes[0] == {"dim_0": 0, "map": list(range(5)), "reduce": ["a", "b"]}
        graph = cas.graph()
        assert len(graph.sinks) == 5
        serialise(graph)

    @pytest.mark.parametrize(
        "args, expected_qube_or_error",
        [
            [["new_dim", "x"], {"dim_0": [0], "dim_1": [0, 1, 2, 3], "new_dim": ["x"]}],
            [["dim_0", 2], ValueError],
            [["dim_0", 2, True], {"dim_0": [2], "dim_1": [0, 1, 2, 3]}],
        ],
        ids=["new-coord", "existing-coord", "override-existing-coord"],
    )
    def test_set_coords(self, args, expected_qube_or_error):
        action = action_from_shape((1, 4))
        if isinstance(expected_qube_or_error, dict):
            new_action = action.add_scalar_dimension(*args)
            for dim, values in expected_qube_or_error.items():
                assert new_action.qube.axes()[dim] == values
        else:
            with pytest.raises(expected_qube_or_error):
                action.add_scalar_dimension(*args)


class TestMultipleDatacubes:
    @pytest.mark.parametrize(
        "actions, datacubes",
        [
            [
                [action_from_shape((3, 4)).add_scalar_dimension("branch", 1), action_from_shape((3, 4)).add_scalar_dimension("branch", 2)],
                Qube.from_datacube({"branch": [1, 2], "dim_0": [0, 1, 2], "dim_1": [0, 1, 2, 3]}),
            ],
            [
                [from_source("test", datacubes={"dim": [0]}), from_source("test", datacubes={"dim": [1]})],
                Qube.from_datacube({"dim": [0, 1]}),
            ],
            [
                [
                    from_source("test", datacubes={"dim": [0], "dim1": [0]}),
                    from_source("test", datacubes={"dim": [1], "dim1": [0]}),
                    from_source("test", datacubes={"dim": [0], "dim1": [1]}),
                    from_source("test", datacubes={"dim": [1], "dim1": [1]}),
                ],
                Qube.from_datacube({"dim": [0, 1], "dim1": [0, 1]}),
            ],
            [
                [
                    from_source("test", datacubes={"branch": 1, "dim": [0], "dim1": [0]}),
                    from_source("test", datacubes={"branch": 1, "dim": [1], "dim1": [0]}),
                    from_source("test", datacubes={"branch": 2, "dim1": [1]}),
                    from_source("test", datacubes={"branch": 2, "dim1": [2]}),
                ],
                Qube.from_datacube({"branch": 1, "dim": [0, 1], "dim1": [0]}) | Qube.from_datacube({"branch": 2, "dim1": [1, 2]}),
            ],
        ],
    )
    def test_merge(self, actions: list[Action], datacubes: Qube):
        output = merge(*actions)
        assert output.qube.to_ascii() == datacubes.to_ascii()

    def test_operation_order(self):
        merged = merge(
            from_source(
                {
                    NodeKey({"branch": 1, "subbranch": branch, "dim_0": x, "dim_1": y}): "func1"
                    for branch in [1, 2]
                    for x in range(3)
                    for y in range(4)
                }
            ),
            from_source(
                {NodeKey({"branch": 2, "dim_0": x, "dim_1": y, "dim_2": z}): "func2" for x in range(5) for y in range(4) for z in range(6)}
            ),
        )
        assert len(merged.qube) == 2

        reduced = merged.select(subbranch=1).flatten(new_dim="temp").concatenate(dim="temp")
        flattened = merged.flatten(new_dim="temp", keep_dims=["branch", "subbranch"]).concatenate(dim="temp").sel(subbranch=1)
        assert len(reduced.nodes) == len(flattened.nodes)
        graph = deduplicate_nodes(reduced.graph() + flattened.graph())
        assert len(graph.sinks) == 1

    @pytest.mark.parametrize(
        "selection, expected_qube_or_error",
        [
            (
                {"dim_0": 1},
                Qube.from_datacube({"dim_0": 1, "dim_1": [0, 1, 2, 3]})
                | Qube.from_datacube({"date": datetime(2024, 1, 1), "dim_0": 1, "dim_1": [0, 1, 2, 3, 4]}),
            ),
            ({"dim_1": 4}, Qube.from_datacube({"date": datetime(2024, 1, 1), "dim_0": [0, 1], "dim_1": 4})),
            (
                {"date": [datetime(2024, 1, 1)]},
                Qube.from_datacube({"date": datetime(2024, 1, 1), "dim_0": [0, 1], "dim_1": [0, 1, 2, 3, 4]}),
            ),
            ({"dim_1": 10}, ValueError),
            (
                {"dim_0": [1], "dim_1": [0, 4]},
                Qube.from_datacube({"dim_0": 1, "dim_1": 0})
                | Qube.from_datacube({"date": datetime(2024, 1, 1), "dim_0": 1, "dim_1": [0, 4]}),
            ),
            (
                {"dim_0": [2], "dim_1": [0, 4]},
                ValueError,
            ),
        ],
    )
    def test_select(self, selection: dict[str, Any], expected_qube_or_error):
        action = merge(
            from_source("test", datacubes={"dim_0": [0, 1, 2], "dim_1": [0, 1, 2, 3]}),
            from_source("test", datacubes={"date": datetime(2024, 1, 1), "dim_0": [0, 1], "dim_1": [0, 1, 2, 3, 4]}),
        )
        if isinstance(expected_qube_or_error, Qube):
            select_dim = action.sel(selection)
            assert select_dim.qube.to_ascii() == expected_qube_or_error.to_ascii()
        else:
            with pytest.raises(expected_qube_or_error, match="No nodes found matching selection criteria:"):
                action.sel(selection).qube
