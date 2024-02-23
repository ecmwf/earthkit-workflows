import os
import numpy as np
import xarray as xr
import pytest

from cascade.executors.dask import DaskLocalExecutor
from cascade.fluent import Fluent, Payload
from cascade.backends.arrayapi import ArrayApiBackend
from cascade.backends.xarray import XArrayBackend


def graph(backend, func):
    args = [np.fromiter([(2, 3) for _ in range(4)], dtype=object) for _ in range(5)]
    return (
        Fluent(backend=backend)
        .source(func, xr.DataArray(args, dims=["x", "y"]))
        .mean("x")
        .minimum("y")
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


def test_graph_execution_jax(tmpdir):
    jax = pytest.importorskip("jax")
    from cascade.backends.jax import JaxBackend

    g = graph(JaxBackend, lambda *x: jax.numpy.asarray(np.random.rand(*x)))

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskLocalExecutor.execute(g, 2, report=f"{tmpdir}/report.html")
    assert len(output) == 3
    assert list(output.values())[0].shape == (2,)
    assert np.all([isinstance(x, jax.Array) for x in output.values()])
