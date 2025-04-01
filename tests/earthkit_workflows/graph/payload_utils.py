# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Callable

from earthkit.workflows.graph import Graph, Node


def node_number(node: Node, sentinel: object) -> object | int:
    _, s, n = node.name.rpartition("-")
    if not s:
        return sentinel
    try:
        n = int(n)  # type: ignore
    except ValueError:
        return sentinel
    return n


def add_payload(g: Graph, func: Callable[[Node, object], object | Any] = node_number):
    sentinel = object()
    for node in g.nodes():
        pl = func(node, sentinel)
        if pl is sentinel:
            continue
        node.payload = pl
