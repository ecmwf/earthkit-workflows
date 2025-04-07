# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator, Sequence, cast

from earthkit.workflows.graph.nodes import Node


class Graph:
    """Graph class

    A graph is represented by the list of its sinks.

    Parameters
    ----------
    sinks: list[Node]
        Sinks of the graph
    """

    sinks: list[Node]

    def __init__(self, sinks: Sequence[Node]):
        # NOTE we need to cast to support covariance for the fluent.Node. Fix hierarchy instead
        self.sinks = cast(list[Node], sinks)
        # if any(culprit := e for e in sinks if not e.is_sink()):
        #     # NOTE consider creating a sink view instead
        #     raise ValueError(f"not a sink: {culprit}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return NotImplemented

        nodes = {n.name: n for n in self.nodes()}
        other_nodes = {n.name: n for n in other.nodes()}
        if nodes.keys() != other_nodes.keys():
            return False
        for name, node in nodes.items():
            onode = other_nodes[name]
            if node.name != onode.name:
                return False
            if node.outputs != onode.outputs:
                return False
            if node.inputs.keys() != onode.inputs.keys():
                return False
            for iname, src in node.inputs.items():
                osrc = onode.inputs[iname]
                if src.parent.name != osrc.parent.name:
                    return False
                if src.name != osrc.name:
                    return False
            if node.payload != onode.payload:
                return False
        return True

    def __add__(self, other: object) -> "Graph":
        if not isinstance(other, Graph):
            return NotImplemented

        return Graph(self.sinks + other.sinks)

    def __iadd__(self, other: object) -> "Graph":
        if not isinstance(other, Graph):
            return NotImplemented

        self.sinks.extend(other.sinks)
        return self

    @classmethod
    def empty(cls) -> "Graph":
        """Just making this a monoid"""
        return cls(sinks=[])

    def nodes(self, forwards=False) -> Iterator[Node]:
        """Iterate over nodes of the graph

        If ``forwards`` is true, iterate in topological order. Otherwise,
        iterate backwards starting from the sinks.
        """
        done: set[Node] = set()
        todo: list[Node] = [sink for sink in self.sinks]

        while todo:
            node = todo[-1]
            if node in done:
                todo.pop()
                continue

            if not forwards:
                todo.pop()

            complete = True
            for isrc in node.inputs.values():
                inode = isrc.parent
                if inode not in done:
                    todo.append(inode)
                    if forwards:
                        complete = False
                        break

            if not complete:
                continue

            if forwards:
                assert todo[-1] is node
                todo.pop()

            yield node
            done.add(node)

    def sources(self) -> Iterator[Node]:
        """Iterate over the sources in the graph"""
        return (n for n in self.nodes(forwards=True) if n.is_source())

    def get_node(self, name: str) -> Node:
        """Get a node by name

        Raises `KeyError` if not found.
        """
        for node in self.nodes():
            if node.name == name:
                return node
        raise KeyError(name)

    def get_predecessors(self, node: Node) -> dict[str, Node | tuple[Node, str]]:
        """Get the predecessors (parents) of a node

        The result is a dict where keys are the given node's input names, and
        values are node outputs, encoded as either the node itself (default
        output), or (parent, output name) tuples.
        """
        return {
            iname: (
                isrc.parent
                if isrc.name == Node.DEFAULT_OUTPUT
                else (isrc.parent, isrc.name)
            )
            for iname, isrc in node.inputs.items()
        }

    def get_successors(self, node: Node) -> dict[str, list[tuple[Node, str]]]:
        """Get the successors (children) of a node

        The result is a dict where keys are the given node's output names, and
        values are lists of (child, input name) tuples.
        """
        succ: dict[str, list[tuple[Node, str]]] = {}
        for other in self.nodes():
            for iname, isrc in other.inputs.items():
                if isrc.parent is node:
                    assert isrc.name in node.outputs
                    succ.setdefault(isrc.name, []).append((other, iname))
        return succ

    def has_cycle(self) -> bool:
        """Check whether a graph contains cycles"""
        done: set[Node] = set()
        todo: list[tuple[Node, list[Node]]] = [(sink, []) for sink in self.sinks]

        while todo:
            node, path = todo[-1]
            assert node not in done

            complete = True
            newpath = path + [node]
            for isrc in node.inputs.values():
                inode = isrc.parent
                if inode not in done:
                    if inode in newpath:
                        return True
                    todo.append((inode, newpath))
                    complete = False
                    break

            if not complete:
                continue

            assert todo[-1][0] is node
            todo.pop()
            done.add(node)

        return False
