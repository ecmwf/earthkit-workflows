from typing import Any, Callable

from cascade.graph import Graph, Node


def node_number(node: Node, sentinel: object) -> object | int:
    _, s, n = node.name.rpartition("-")
    if not s:
        return sentinel
    try:
        n = int(n) # type: ignore
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
