import pytest
import numpy as np

from cascade.functions import multi_arg_function, two_arg_function


def inputs(number: int, shape=(2, 3)):
    return [np.random.rand(*shape) for _ in range(number)]


@pytest.mark.parametrize(
    ["num_inputs", "kwargs", "output_shape"],
    [[4, {}, (2, 3)], [1, {"axis": 0}, (3,)], [1, {"axis": 1}, (2,)]],
)
def test_multi_arg(num_inputs, kwargs, output_shape):
    assert (
        multi_arg_function("mean", *inputs(num_inputs), **kwargs).shape == output_shape
    )


@pytest.mark.parametrize(
    ["num_inputs", "output_shape"],
    [
        [2, (2, 3)],
        [1, (3,)],
    ],
)
def test_two_arg(num_inputs, output_shape):
    assert two_arg_function("add", *inputs(num_inputs)).shape == output_shape


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
