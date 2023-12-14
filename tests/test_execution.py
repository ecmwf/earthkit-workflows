import os
import numpy as np
import xarray as xr
import pytest

from cascade.executor import DaskExecutor
from cascade.scheduler import Schedule
from cascade.fluent import Fluent, Payload

input = np.random.rand(2, 3)


def graph(payloads):
    return (
        Fluent.source(payloads, ["x", "y"])
        .mean("x")
        .minimum("y")
        .expand("z", 3, 1, 0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )


@pytest.mark.parametrize(
    "payload, output_type",
    [
        [Payload(np.asarray, [input]), np.ndarray],
        [Payload(xr.DataArray, [input], {"dims": ["a", "b"]}), xr.DataArray],
    ],
)
def test_graph_execution(payload, output_type):
    payloads = np.empty((4, 5), dtype=object)
    payloads[:] = payload
    g = graph(payloads)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    schedule = Schedule(g, None, {})
    executor = DaskExecutor(schedule)
    output = executor.execute(memory_limit="10GB", n_workers=1, threads_per_worker=1)
    assert len(output) == 3
    assert output[0].shape == (2,)
    assert np.all([isinstance(x, output_type) for x in output])


def test_graph_execution_jax():
    jax = pytest.importorskip("jax")
    payloads = np.empty((4, 5), dtype=object)
    payloads[:] = Payload(jax.numpy.asarray, [input])
    g = graph(payloads)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    schedule = Schedule(g, None, {})
    executor = DaskExecutor(schedule)
    output = executor.execute(memory_limit="10GB", n_workers=1, threads_per_worker=1)
    assert len(output) == 3
    assert output[0].shape == (2,)
    assert np.all([isinstance(x, jax.Array) for x in output])
