# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import xarray as xr

from earthkit.workflows.fluent import Action, Node, Payload


class MockNode(Node):
    def __init__(self, name: str):
        super().__init__(Payload(lambda name=name: name))


def mock_action(shape: tuple) -> Action:
    nodes = np.empty(shape, dtype=object)
    it = np.nditer(nodes, flags=["multi_index", "refs_ok"])
    for _ in it:
        nodes[it.multi_index] = MockNode(f"{it.multi_index}")
    nodes_xr = xr.DataArray(
        nodes, coords={f"dim_{x}": list(range(dim)) for x, dim in enumerate(shape)}
    )
    return Action(nodes_xr)
