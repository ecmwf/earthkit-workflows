import os
import numpy as np
import xarray as xr
import pytest
import functools

from cascade.executors.dask import DaskLocalExecutor
from cascade.fluent import from_source
from cascade.profiler import profile

from helpers import mock_graph


@pytest.mark.parametrize(
    "func, output_type",
    [
        [functools.partial(np.random.rand, 100, 100), np.ndarray],
        [
            lambda x=[100, 100]: xr.DataArray(
                np.random.rand(*x), dims=[f"x{i}" for i in range(len(x))]
            ),
            xr.DataArray,
        ],
    ],
)
def test_graph_execution(tmpdir, func, output_type):
    g = mock_graph(func)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskLocalExecutor().execute(g, report=f"{tmpdir}/report.html")
    assert len(output) == 3
    assert list(output.values())[0].shape == (100,)
    assert np.all([isinstance(x, output_type) for x in output.values()])


def test_graph_benchmark(tmpdir):
    g = mock_graph(functools.partial(np.random.rand, 100, 100))

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    executor = DaskLocalExecutor(n_workers=2)
    _, annotated_graph = profile(g, tmpdir, executor)
    nodes = list(annotated_graph.nodes())
    assert not all([node.cost == 0 for node in nodes])
    assert not all([node.memory == 0 for node in nodes])


@pytest.mark.skip("Need new Array API Compat release with JAX helpers")
def test_graph_execution_jax(tmpdir):
    jax = pytest.importorskip("jax")

    g = mock_graph(lambda *x: jax.numpy.asarray(np.random.rand(*x)))

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskLocalExecutor().execute(g, 2, report=f"{tmpdir}/report.html")
    assert len(output) == 3
    assert list(output.values())[0].shape == (2,)
    assert np.all([isinstance(x, jax.Array) for x in output.values()])


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
    sources = from_source(funcs, ["x", "y"])

    non_batched = getattr(sources, func)("x")
    assert len(list(non_batched.graph().nodes())) == 24
    batched = getattr(sources, func)("x", batch_size=3)
    assert len(list(batched.graph().nodes())) >= 32
    g = non_batched.subtract(batched).graph()
    output = DaskLocalExecutor(n_workers=2).execute(g, report=f"{tmpdir}/report.html")
    print(func, output)
    for value in output.values():
        assert np.all(value == 0)
