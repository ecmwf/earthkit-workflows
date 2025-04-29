# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from earthkit.workflows.graph import Node, join_namespaced, rename_nodes, serialise
from earthkit.workflows.graph.samplegraphs import disconnected, linear, multi, simple

D = Node.DEFAULT_OUTPUT


def test_rename():
    g = multi(5, 3, 2)
    gr = rename_nodes((lambda name: "r." + name), g)
    assert serialise(gr) == {
        "r.reader-0": {"inputs": {}, "outputs": [D]},
        "r.reader-1": {"inputs": {}, "outputs": [D]},
        "r.reader-2": {"inputs": {}, "outputs": [D]},
        "r.reader-3": {"inputs": {}, "outputs": [D]},
        "r.reader-4": {"inputs": {}, "outputs": [D]},
        "r.process-0": {
            "inputs": {f"input{i}": f"r.reader-{i}" for i in range(5)},
            "outputs": ["output0", "output1", "output2"],
        },
        "r.process-1": {
            "inputs": {
                "input1": ("r.process-0", "output0"),
                "input2": ("r.process-0", "output2"),
            },
            "outputs": [D],
        },
        "r.process-2": {
            "inputs": {"input1": ("r.process-0", "output1"), "input2": ("r.reader-2")},
            "outputs": ["output0", "output1"],
        },
        "r.writer-0": {
            "inputs": {"input1": "r.process-1", "input2": ("r.process-2", "output0")},
            "outputs": [],
        },
        "r.writer-1": {
            "inputs": {"input1": "r.process-1", "input2": ("r.process-2", "output1")},
            "outputs": [],
        },
    }


def test_join_namespaced():
    g = join_namespaced(g1=linear(2), g2=disconnected(2), g3=simple(3, 2))
    assert serialise(g) == {
        "g1.reader": {"inputs": {}, "outputs": [D]},
        "g1.process-0": {"inputs": {"input": "g1.reader"}, "outputs": [D]},
        "g1.process-1": {"inputs": {"input": "g1.process-0"}, "outputs": [D]},
        "g1.writer": {"inputs": {"input": "g1.process-1"}, "outputs": []},
        "g2.reader-0": {"inputs": {}, "outputs": [D]},
        "g2.process-0": {"inputs": {"input": "g2.reader-0"}, "outputs": [D]},
        "g2.writer-0": {"inputs": {"input": "g2.process-0"}, "outputs": []},
        "g2.reader-1": {"inputs": {}, "outputs": [D]},
        "g2.process-1": {"inputs": {"input": "g2.reader-1"}, "outputs": [D]},
        "g2.writer-1": {"inputs": {"input": "g2.process-1"}, "outputs": []},
        "g3.reader-0": {"inputs": {}, "outputs": [D]},
        "g3.reader-1": {"inputs": {}, "outputs": [D]},
        "g3.reader-2": {"inputs": {}, "outputs": [D]},
        "g3.process-0": {
            "inputs": {
                "input0": "g3.reader-0",
                "input1": "g3.reader-1",
                "input2": "g3.reader-2",
            },
            "outputs": [D],
        },
        "g3.process-1": {
            "inputs": {
                "input0": "g3.reader-0",
                "input1": "g3.reader-1",
                "input2": "g3.reader-2",
            },
            "outputs": [D],
        },
        "g3.writer-0": {"inputs": {"input": "g3.process-0"}, "outputs": []},
        "g3.writer-1": {"inputs": {"input": "g3.process-1"}, "outputs": []},
    }
