import os
import numpy as np
import xarray as xr
import pytest

from cascade.executors.dask import DaskLocalExecutor
from cascade.fluent import Fluent, Payload
from cascade.backends.arrayapi import ArrayApiBackend
from cascade.backends.xarray import XArrayBackend
from cascade.graph import deduplicate_nodes


def graph(backend, func):
    args = [np.fromiter([(2, 3) for _ in range(4)], dtype=object) for _ in range(5)]
    return (
        Fluent(backend=backend)
        .source(func, xr.DataArray(args, dims=["x", "y"]))
        .mean("x")
        .min("y")
        .expand("z", 3, 1, 0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )


@pytest.mark.parametrize(
    "backend, func, output_type",
    [
        [ArrayApiBackend, np.random.rand, np.ndarray],
        [
            XArrayBackend,
            lambda *x: xr.DataArray(
                np.random.rand(*x), dims=[f"x{i}" for i in range(len(x))]
            ),
            xr.DataArray,
        ],
    ],
)
def test_graph_execution(tmpdir, backend, func, output_type):
    g = graph(backend, func)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskLocalExecutor.execute(g, report=f"{tmpdir}/report.html")
    assert len(output) == 3
    assert list(output.values())[0].shape == (2,)
    assert np.all([isinstance(x, output_type) for x in output.values()])


@pytest.mark.skip("Need new Array API Compat release with JAX helpers")
def test_graph_execution_jax(tmpdir):
    jax = pytest.importorskip("jax")

    g = graph(ArrayApiBackend, lambda *x: jax.numpy.asarray(np.random.rand(*x)))

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskLocalExecutor.execute(g, 2, report=f"{tmpdir}/report.html")
    assert len(output) == 3
    assert list(output.values())[0].shape == (2,)
    assert np.all([isinstance(x, jax.Array) for x in output.values()])


@pytest.mark.parametrize(
    "func", ["mean", "std", "min", "max", "sum", "prod", "concatenate"]
)
def test_batch_execution(tmpdir, func):
    args = [np.fromiter([(1, 100) for _ in range(4)], dtype=object) for _ in range(5)]
    sources = Fluent(backend=ArrayApiBackend).source(
        np.random.randint, xr.DataArray(args, dims=["x", "y"]), {"size": (2, 3)}
    )

    non_batched = getattr(sources, func)("x")
    assert len(list(non_batched.graph().nodes())) == 24
    batched = getattr(sources, func)("x", batch_size=3)
    assert len(list(batched.graph().nodes())) >= 32
    g = non_batched.subtract(batched).graph()
    output = DaskLocalExecutor.execute(g, 2, report=f"{tmpdir}/report.html")
    print(func, output)
    for value in output.values():
        assert np.all(value == 0)
