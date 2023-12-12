import pytest
import numpy as np
import xarray as xr

from cascade import backends


def inputs(number: int, shape=(2, 3)):
    return [
        xr.DataArray(
            np.random.rand(*shape), dims=[f"dim{x}" for x in range(len(shape))]
        )
        for _ in range(number)
    ]


@pytest.mark.parametrize(
    ["num_inputs", "kwargs", "output_shape"],
    [
        [4, {}, (2, 3)],
        [1, {"method_kwargs": {"dim": "dim0"}}, (3,)],
        [1, {"method_kwargs": {"dim": "dim1"}}, (2,)],
    ],
)
def test_multi_arg(num_inputs, kwargs, output_shape):
    assert backends.mean(*inputs(num_inputs), **kwargs).shape == output_shape


@pytest.mark.parametrize(
    ["num_inputs", "output_shape"],
    [
        [2, (2, 3)],
    ],
)
def test_two_arg(num_inputs, output_shape):
    assert backends.add(*inputs(num_inputs)).shape == output_shape


@pytest.mark.parametrize(
    ["num_inputs", "shape"],
    [
        [3, (2, 3)],
        [1, (3,)],
    ],
)
def test_two_arg_raises(num_inputs, shape):
    with pytest.raises(Exception):
        backends.add(*inputs(num_inputs, shape))


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[[0]], {"axis": 0}, (1, 3)],
        [[[0, 1]], {"axis": 1}, (2, 2)],
    ],
)
def test_single_arg(args, kwargs, output_shape):
    output = backends.take(*inputs(1), *args, **kwargs)
    assert output.shape == output_shape


def test_concatenate():
    input = inputs(3) + inputs(2, (2,))

    # Without dim
    with pytest.raises(Exception):
        backends.concatenate(*input)

    # With dim
    backends.concatenate(*input, dim="dim1")


def test_stack():
    input = inputs(3) + inputs(2, (2,))

    x = backends.stack(*input, dim="NEW")

    # Without dim
    with pytest.raises(Exception):
        backends.stack(*input)

    # With existing dim
    with pytest.raises(Exception):
        backends.stack(*input, dim="dim0")

    # With dim and axis
    y = backends.stack(*input, axis=2, dim="NEW")
    assert np.all(x.transpose("dim0", "dim1", "NEW") == y)
