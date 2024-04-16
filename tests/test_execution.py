import os
import numpy as np
import xarray as xr
import pytest

from cascade.executors.dask import DaskLocalExecutor
from cascade.fluent import Fluent

from helpers import mock_graph


@pytest.mark.parametrize(
    "func, output_type",
    [
        [np.random.rand, np.ndarray],
        [
            lambda *x: xr.DataArray(
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
    g = mock_graph(np.random.rand)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    resource_map = DaskLocalExecutor(n_workers=2).benchmark(
        g, report=f"{tmpdir}/report.html", mem_report=f"{tmpdir}/mem.csv"
    )
    assert np.all([x.name in resource_map for x in g.nodes()])
    resources = np.asarray(
        [[resource_map[x.name].cost, resource_map[x.name].memory] for x in g.nodes()]
    )
    assert not np.all(resources[:, 0] == 0)
    assert not np.all(resources[:, 1] == 0)


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
    args = [np.fromiter([(1, 100) for _ in range(4)], dtype=object) for _ in range(5)]
    sources = Fluent().source(
        np.random.randint, xr.DataArray(args, dims=["x", "y"]), {"size": (2, 3)}
    )

    non_batched = getattr(sources, func)("x")
    assert len(list(non_batched.graph().nodes())) == 24
    batched = getattr(sources, func)("x", batch_size=3)
    assert len(list(batched.graph().nodes())) >= 32
    g = non_batched.subtract(batched).graph()
    output = DaskLocalExecutor(n_workers=2).execute(g, report=f"{tmpdir}/report.html")
    print(func, output)
    for value in output.values():
        assert np.all(value == 0)
