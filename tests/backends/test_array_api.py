import pytest
import numpy as np

from cascade.backends.array_api import (
    multi_arg_function,
    two_arg_function,
    single_arg_function,
)


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
        multi_arg_function("mean", *inputs(num_inputs, input_shape), **kwargs).shape
        == output_shape
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
    assert (
        two_arg_function("add", *inputs(num_inputs, input_shape)).shape == output_shape
    )


@pytest.mark.parametrize(
    ["num_inputs", "shape"],
    [
        [3, (2, 3)],
        [1, (3,)],
    ],
)
def test_two_arg_raises(num_inputs, shape):
    with pytest.raises(Exception):
        two_arg_function("add", *inputs(num_inputs, shape))


@pytest.mark.parametrize(
    ["name", "num_inputs", "kwargs", "output_shape"],
    [
        ["take", 1, {"indices": [0], "axis": 0}, (1, 3)],
        ["take", 1, {"indices": [0, 1], "axis": 1}, (2, 2)],
    ],
)
def test_single_arg(name, num_inputs, kwargs, output_shape):
    output = single_arg_function(name, *inputs(num_inputs), **kwargs)
    assert output.shape == output_shape
