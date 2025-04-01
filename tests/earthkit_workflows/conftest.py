import functools

import numpy as np
import pytest

from earthkit.workflows.fluent import Payload, from_source


@pytest.fixture(scope="function")
def task_graph(request):
    func = getattr(request, "param", functools.partial(np.random.rand, 2, 3))
    return (
        from_source(
            [
                np.fromiter(
                    [func for _ in range(6)],
                    dtype=object,
                )
                for _ in range(7)
            ],
            dims=["x", "y"],
        )
        .mean("x")
        .min("y")
        .expand("z", internal_dim=1, dim_size=3, axis=0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )
