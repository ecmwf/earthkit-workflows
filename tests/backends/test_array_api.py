import pytest
import numpy as np

from cascade.backends.array_api import ArrayApiBackend

from generic_tests import *


def inputs(num_inputs: int, input_shape=(2, 3)):
    return [np.random.rand(*input_shape) for _ in range(num_inputs)]


@pytest.fixture
def backend():
    return ArrayApiBackend


@pytest.fixture
def input_generator():
    return inputs


def test_instantiation():
    ArrayApiBackend()


@pytest.mark.parametrize(
    ["num_inputs", "input_shape", "kwargs", "output_shape"],
    [
        [1, (2, 3), {"axis": 0}, (3,)],
        [1, (2, 3), {"axis": 1}, (2,)],
    ],
)
def test_multi_arg_axis(num_inputs, input_shape, kwargs, output_shape):
    for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
        assert (
            getattr(ArrayApiBackend, func)(
                *inputs(num_inputs, input_shape), **kwargs
            ).shape
            == output_shape
        )


@pytest.mark.parametrize(
    ["num_inputs", "input_shape", "output_shape"],
    [
        [1, (2, 3), (3,)],
        [1, (2,), ()],
    ],
)
def test_two_arg_single(input_generator, num_inputs, input_shape, output_shape):
    for func in ["add", "subtract", "multiply", "divide"]:
        assert (
            getattr(ArrayApiBackend, func)(
                *input_generator(num_inputs, input_shape)
            ).shape
            == output_shape
        )


def test_concatenate():
    input = inputs(3) + inputs(2, (2, 1))

    # Without axis
    with pytest.raises(Exception):
        ArrayApiBackend.concat(*input)

    # With axis
    assert ArrayApiBackend.concat(*input, axis=1).shape == (2, 11)
    assert ArrayApiBackend.concat(*inputs(1), axis=1).shape == (2, 3)


def test_stack():
    input = inputs(3) + inputs(2, (2, 1))

    assert ArrayApiBackend.stack(*input, axis=0).shape == (5, 2, 3)
    assert ArrayApiBackend.stack(*inputs(1), axis=0).shape == (1, 2, 3)
