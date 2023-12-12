import pytest
import numpy as np

from cascade import backends


def inputs(number: int, shape=(2, 3)):
    return [np.random.rand(*shape) for _ in range(number)]


@pytest.mark.parametrize(
    ["num_inputs", "input_shape", "kwargs", "output_shape"],
    [
        [4, (2, 3), {}, (2, 3)],
        [1, (2, 3), {"axis": 0}, (3,)],
        [1, (2, 3), {"axis": 1}, (2,)],
        [1, (2, 3), {}, ()],
    ],
)
def test_multi_arg(num_inputs, input_shape, kwargs, output_shape):
    assert (
        backends.mean(*inputs(num_inputs, input_shape), **kwargs).shape == output_shape
    )


@pytest.mark.parametrize(
    ["num_inputs", "input_shape", "output_shape"],
    [
        [2, (2, 3), (2, 3)],
        [1, (2, 3), (3,)],
        [1, (2,), ()],
    ],
)
def test_two_arg(num_inputs, input_shape, output_shape):
    assert backends.add(*inputs(num_inputs, input_shape)).shape == output_shape


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
