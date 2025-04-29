# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from payload_utils import add_payload

from earthkit.workflows.graph import Node, deserialise, serialise
from earthkit.workflows.graph.samplegraphs import empty, linear, multi, simple

D = Node.DEFAULT_OUTPUT


def test_serialise_empty():
    g = empty()
    s = serialise(g)
    assert s == {}


def test_serialise_linear():
    g = linear(5)
    s = serialise(g)
    assert s == {
        "reader": {"inputs": {}, "outputs": [D]},
        "process-0": {"inputs": {"input": "reader"}, "outputs": [D]},
        "process-1": {"inputs": {"input": "process-0"}, "outputs": [D]},
        "process-2": {"inputs": {"input": "process-1"}, "outputs": [D]},
        "process-3": {"inputs": {"input": "process-2"}, "outputs": [D]},
        "process-4": {"inputs": {"input": "process-3"}, "outputs": [D]},
        "writer": {"inputs": {"input": "process-4"}, "outputs": []},
    }


def test_serialise_simple():
    g = simple(5, 3)
    s = serialise(g)
    pi = {f"input{i}": f"reader-{i}" for i in range(5)}
    assert s == {
        "reader-0": {"inputs": {}, "outputs": [D]},
        "reader-1": {"inputs": {}, "outputs": [D]},
        "reader-2": {"inputs": {}, "outputs": [D]},
        "reader-3": {"inputs": {}, "outputs": [D]},
        "reader-4": {"inputs": {}, "outputs": [D]},
        "process-0": {"inputs": pi, "outputs": [D]},
        "process-1": {"inputs": pi, "outputs": [D]},
        "process-2": {"inputs": pi, "outputs": [D]},
        "writer-0": {"inputs": {"input": "process-0"}, "outputs": []},
        "writer-1": {"inputs": {"input": "process-1"}, "outputs": []},
        "writer-2": {"inputs": {"input": "process-2"}, "outputs": []},
    }


def test_serialise_multi():
    g = multi(5, 3, 2)
    s = serialise(g)
    assert s == {
        "reader-0": {"inputs": {}, "outputs": [D]},
        "reader-1": {"inputs": {}, "outputs": [D]},
        "reader-2": {"inputs": {}, "outputs": [D]},
        "reader-3": {"inputs": {}, "outputs": [D]},
        "reader-4": {"inputs": {}, "outputs": [D]},
        "process-0": {
            "inputs": {f"input{i}": f"reader-{i}" for i in range(5)},
            "outputs": ["output0", "output1", "output2"],
        },
        "process-1": {
            "inputs": {
                "input1": ("process-0", "output0"),
                "input2": ("process-0", "output2"),
            },
            "outputs": [D],
        },
        "process-2": {
            "inputs": {"input1": ("process-0", "output1"), "input2": ("reader-2")},
            "outputs": ["output0", "output1"],
        },
        "writer-0": {
            "inputs": {"input1": "process-1", "input2": ("process-2", "output0")},
            "outputs": [],
        },
        "writer-1": {
            "inputs": {"input1": "process-1", "input2": ("process-2", "output1")},
            "outputs": [],
        },
    }


def test_deserialise_empty():
    gs = {}
    g = deserialise(gs)
    assert g == empty()


def test_deserialise_linear():
    gs = {
        "reader": {"inputs": {}, "outputs": [D]},
        "process-0": {"inputs": {"input": "reader"}, "outputs": [D]},
        "process-1": {"inputs": {"input": "process-0"}, "outputs": [D]},
        "process-2": {"inputs": {"input": "process-1"}, "outputs": [D]},
        "process-3": {"inputs": {"input": "process-2"}, "outputs": [D]},
        "writer": {"inputs": {"input": "process-3"}, "outputs": []},
    }
    g = deserialise(gs)
    assert g == linear(4)


def test_deserialise_simple():
    pi = {f"input{i}": f"reader-{i}" for i in range(4)}
    gs = {
        "reader-0": {"inputs": {}, "outputs": [D]},
        "reader-1": {"inputs": {}, "outputs": [D]},
        "reader-2": {"inputs": {}, "outputs": [D]},
        "reader-3": {"inputs": {}, "outputs": [D]},
        "process-0": {"inputs": pi, "outputs": [D]},
        "process-1": {"inputs": pi, "outputs": [D]},
        "writer-0": {"inputs": {"input": "process-0"}, "outputs": []},
        "writer-1": {"inputs": {"input": "process-1"}, "outputs": []},
    }
    g = deserialise(gs)
    assert g == simple(4, 2)


def test_deserialise_multi():
    gs = {
        "reader-0": {"inputs": {}, "outputs": [D]},
        "reader-1": {"inputs": {}, "outputs": [D]},
        "reader-2": {"inputs": {}, "outputs": [D]},
        "reader-3": {"inputs": {}, "outputs": [D]},
        "process-0": {
            "inputs": {f"input{i}": f"reader-{i}" for i in range(4)},
            "outputs": ["output0", "output1", "output2", "output3"],
        },
        "process-1": {
            "inputs": {
                "input1": ("process-0", "output0"),
                "input2": ("process-0", "output2"),
            },
            "outputs": [D],
        },
        "process-2": {
            "inputs": {
                "input1": ("process-0", "output0"),
                "input2": ("process-0", "output3"),
            },
            "outputs": [D],
        },
        "process-3": {
            "inputs": {"input1": ("process-0", "output1"), "input2": ("reader-2")},
            "outputs": ["output0", "output1"],
        },
        "writer-0": {
            "inputs": {"input1": "process-1", "input2": ("process-3", "output0")},
            "outputs": [],
        },
        "writer-1": {
            "inputs": {"input1": "process-2", "input2": ("process-3", "output0")},
            "outputs": [],
        },
        "writer-2": {
            "inputs": {"input1": "process-1", "input2": ("process-3", "output1")},
            "outputs": [],
        },
        "writer-3": {
            "inputs": {"input1": "process-2", "input2": ("process-3", "output1")},
            "outputs": [],
        },
    }
    g = deserialise(gs)
    assert g == multi(4, 4, 2)


def test_serialise_payload():
    g = linear(4)
    add_payload(g)
    s = serialise(g)
    assert s == {
        "reader": {"inputs": {}, "outputs": [D]},
        "process-0": {"inputs": {"input": "reader"}, "outputs": [D], "payload": 0},
        "process-1": {"inputs": {"input": "process-0"}, "outputs": [D], "payload": 1},
        "process-2": {"inputs": {"input": "process-1"}, "outputs": [D], "payload": 2},
        "process-3": {"inputs": {"input": "process-2"}, "outputs": [D], "payload": 3},
        "writer": {"inputs": {"input": "process-3"}, "outputs": []},
    }


def test_deserialise_payload():
    pi = {f"input{i}": f"reader-{i}" for i in range(5)}
    gs = {
        "reader-0": {"inputs": {}, "outputs": [D], "payload": 0},
        "reader-1": {"inputs": {}, "outputs": [D], "payload": 1},
        "reader-2": {"inputs": {}, "outputs": [D], "payload": 2},
        "reader-3": {"inputs": {}, "outputs": [D], "payload": 3},
        "reader-4": {"inputs": {}, "outputs": [D], "payload": 4},
        "process-0": {"inputs": pi, "outputs": [D], "payload": 0},
        "process-1": {"inputs": pi, "outputs": [D], "payload": 1},
        "process-2": {"inputs": pi, "outputs": [D], "payload": 2},
        "writer-0": {"inputs": {"input": "process-0"}, "outputs": [], "payload": 0},
        "writer-1": {"inputs": {"input": "process-1"}, "outputs": [], "payload": 1},
        "writer-2": {"inputs": {"input": "process-2"}, "outputs": [], "payload": 2},
    }
    g = deserialise(gs)
    exp = simple(5, 3)
    add_payload(exp)
    assert g == exp
