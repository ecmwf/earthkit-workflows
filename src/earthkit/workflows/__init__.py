# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dill

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    # assuming editable install etc
    pass
from . import fluent, mark
from .graph import Graph, deduplicate_nodes
from .graph.export import deserialise, serialise
from .visualise import visualise


class Cascade:
    def __init__(self, graph: Graph = Graph([])):
        self._graph = graph

    @classmethod
    def from_actions(cls, actions):
        graph = Graph([])
        for action in actions:
            graph += action.graph()
        return cls(deduplicate_nodes(graph))

    @classmethod
    def from_serialised(cls, filename: str):
        with open(filename, "rb") as f:
            data = dill.load(f)
            return cls(deserialise(data))

    def serialise(self, filename: str):
        data = serialise(self._graph)
        with open(filename, "wb") as f:
            dill.dump(data, f)

    def visualise(self, *args, **kwargs):
        return visualise(self._graph, *args, **kwargs)

    def __add__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        return Cascade(deduplicate_nodes(self._graph + other._graph))

    def __iadd__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        self._graph += other._graph
        self._graph = deduplicate_nodes(self._graph)
        return self


__all__ = [
    "mark",
    "fluent",
    "Cascade",
]
