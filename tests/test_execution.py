import os
import numpy as np
import xarray as xr
import pytest

from cascade.executors.dask import DaskExecutor
from cascade.fluent import Fluent, Payload
from cascade.backends.arrayapi import ArrayApiBackend
from cascade.backends.xarray import XArrayBackend

input = np.random.rand(2, 3)


def graph(backend, payloads):
    return (
        Fluent(backend=backend)
        .source(payloads, ["x", "y"])
        .mean("x")
        .minimum("y")
        .expand("z", 3, 1, 0)
        .map([Payload(lambda x, a=a: x * a) for a in range(1, 4)])
        .graph()
    )


@pytest.mark.parametrize(
    "backend, payload, output_type",
    [
        [ArrayApiBackend, Payload(np.asarray, [input]), np.ndarray],
        [
            XArrayBackend,
            Payload(xr.DataArray, [input], {"dims": ["a", "b"]}),
            xr.DataArray,
        ],
    ],
)
def test_graph_execution(backend, payload, output_type):
    payloads = np.empty((4, 5), dtype=object)
    payloads[:] = payload
    g = graph(backend, payloads)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskExecutor.execute(g)
    assert len(output) == 3
    assert output[0].shape == (2,)
    assert np.all([isinstance(x, output_type) for x in output])


def test_graph_execution_jax():
    jax = pytest.importorskip("jax")
    from cascade.backends.jax import JaxBackend

    payloads = np.empty((4, 5), dtype=object)
    payloads[:] = Payload(jax.numpy.asarray, [input])
    g = graph(JaxBackend, payloads)

    os.environ["DASK_LOGGING__DISTRIBUTED"] = "debug"
    output = DaskExecutor.execute(g)
    assert len(output) == 3
    assert output[0].shape == (2,)
    assert np.all([isinstance(x, jax.Array) for x in output])
