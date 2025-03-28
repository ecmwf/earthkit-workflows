import functools
import os

import numpy as np
import pytest
import xarray as xr

from cascade.fluent import from_source
from cascade.v0_executors.dask import DaskLocalExecutor


@pytest.mark.parametrize(
    "task_graph, output_type",
    [
        [functools.partial(np.random.rand, 100, 100), np.ndarray],
        [
            lambda x=[100, 100]: xr.DataArray(
                np.random.rand(*x), dims=[f"x{i}" for i in range(len(x))]
            ),
            xr.DataArray,
        ],
    ],
    indirect=["task_graph"],
)
def test_graph_execution(tmpdir, task_graph, output_type):
    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskLocalExecutor().execute(task_graph, report=f"{tmpdir}/report.html")
    assert len(output) == 3
    assert list(output.values())[0].shape == (100,)
    assert np.all([isinstance(x, output_type) for x in output.values()])


@pytest.mark.parametrize(
    "func", ["mean", "std", "min", "max", "sum", "prod", "concatenate"]
)
def test_batch_execution(tmpdir, func):
    funcs = [
        np.fromiter(
            [
                functools.partial(np.random.randint, 1, 100, size=(2, 3))
                for _ in range(4)
            ],
            dtype=object,
        )
        for _ in range(5)
    ]
    sources = from_source(funcs, dims=["x", "y"])

    non_batched = getattr(sources, func)("x")
    assert len(list(non_batched.graph().nodes())) == 24
    batched = getattr(sources, func)("x", batch_size=3)
    assert len(list(batched.graph().nodes())) >= 32
    g = non_batched.subtract(batched).graph()
    output = DaskLocalExecutor(n_workers=2).execute(g, report=f"{tmpdir}/report.html")
    print(func, output)
    for value in output.values():
        assert np.all(value == 0)
