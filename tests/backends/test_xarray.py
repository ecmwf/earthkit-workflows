import pytest
import numpy as np
import xarray as xr

from cascade.backends.xarray import XArrayBackend

from generic_tests import *


def inputs(number: int, shape=(2, 3)):
    return [
        xr.DataArray(
            np.random.rand(*shape), dims=[f"dim{x}" for x in range(len(shape))]
        )
        for _ in range(number)
    ]


@pytest.fixture
def backend():
    return XArrayBackend


@pytest.fixture
def input_generator():
    return inputs


def test_instantiation():
    XArrayBackend()


@pytest.mark.parametrize(
    ["num_inputs", "kwargs", "output_shape"],
    [
        [1, {"dim": "dim0"}, (3,)],
        [1, {"dim": "dim1"}, (2,)],
    ],
)
def test_multi_arg_dim(num_inputs, kwargs, output_shape):
    for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
        assert (
            getattr(XArrayBackend, func)(*inputs(num_inputs), **kwargs).shape
            == output_shape
        )


def test_concatenate():
    # Note broadcasting in xarray works different to numpy,
    # where an array with shape (2, 1) can not be boradcasted to (2, 3)
    # whereas an array with a missing dimension e.g. (2,) can be broadcasted to
    # shape (2, 3)
    input = inputs(3) + inputs(2, (2,))

    # Without dim
    with pytest.raises(Exception):
        XArrayBackend.concat(*input)

    # With dim
    assert XArrayBackend.concat(*input, dim="dim1").shape == (2, 11)
    assert XArrayBackend.concat(*inputs(1), dim="dim1").shape == (2, 3)


def test_stack():
    input = inputs(3) + inputs(2, (2,))

    x = XArrayBackend.stack(*input, dim="NEW")
    assert x.shape == (5, 2, 3)
    assert not np.any(np.isnan(x))

    # Without dim
    with pytest.raises(Exception):
        XArrayBackend.stack(*input)

    # With existing dim
    with pytest.raises(Exception):
        XArrayBackend.stack(*input, dim="dim0")

    # With dim and axis
    y = XArrayBackend.stack(*input, axis=2, dim="NEW")
    assert np.all(x.transpose("dim0", "dim1", "NEW") == y)
    assert XArrayBackend.stack(*inputs(1), axis=0, dim="NEW").shape == (1, 2, 3)
