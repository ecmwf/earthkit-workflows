import numpy as np
import xarray as xr

from cascade.fluent import Action, Node, Payload


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(Payload(name))


def mock_action(shape: tuple) -> Action:
    nodes = np.empty(shape, dtype=object)
    it = np.nditer(nodes, flags=["multi_index", "refs_ok"])
    for _ in it:
        nodes[it.multi_index] = MockNode(f"{it.multi_index}")
    nodes = xr.DataArray(nodes, coords={f"dim_{x}": list(range(dim)) for x, dim in enumerate(shape)})
    return Action(nodes)
