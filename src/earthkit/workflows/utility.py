# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from sortedcontainers import SortedDict

from earthkit.workflows.graph import Graph, Node


class EventLoop:
    def __init__(self):
        self.timesteps = SortedDict()

    def add_event(self, time, callback, *args):
        if time in self.timesteps:
            self.timesteps[time].append((callback, *args))
        else:
            self.timesteps[time] = [(callback, *args)]

    def run(self):
        while len(self.timesteps) > 0:
            time, callbacks = self.timesteps.popitem(0)
            while len(callbacks) > 0:
                callback = callbacks[0]
                callback[0](time, *callback[1:])
                callbacks.pop(0)


def successors(graph: Graph, node: Node) -> list[Node]:
    return [x[0] for x in sum(graph.get_successors(node).values(), [])]


def predecessors(graph: Graph, node: Node) -> list[Node]:
    return [
        x if isinstance(x, Node) else x[0]
        for x in graph.get_predecessors(node).values()
    ]
