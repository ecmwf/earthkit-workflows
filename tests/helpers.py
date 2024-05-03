import numpy as np
import xarray as xr

from cascade.fluent import Action, Node, Payload, from_source


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(Payload(name))


def mock_action(shape: tuple) -> Action:
    nodes = np.empty(shape, dtype=object)
    it = np.nditer(nodes, flags=["multi_index", "refs_ok"])
    for _ in it:
        nodes[it.multi_index] = MockNode(f"{it.multi_index}")
    nodes = xr.DataArray(
        nodes, coords={f"dim_{x}": list(range(dim)) for x, dim in enumerate(shape)}
    )
    return Action(nodes)


def mock_graph(func):
    args = [np.fromiter([func for _ in range(4)], dtype=object) for _ in range(5)]
    return (
        from_source(args, ["x", "y"])
        .mean("x")
        .min("y")
        .expand("z", 3, 1, 0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )
