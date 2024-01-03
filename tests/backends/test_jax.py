import pytest

pytest.importorskip("jax")
import numpy as np
import jax.numpy as jnp

from cascade.backends.jax import JaxBackend

from generic_tests import *


def inputs(number: int, shape=(2, 3)):
    return [jnp.array(np.random.rand(*shape)) for _ in range(number)]


@pytest.fixture
def backend():
    return JaxBackend


@pytest.fixture
def input_generator():
    return inputs


def test_instantiation():
    JaxBackend()


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
            getattr(JaxBackend, func)(*inputs(num_inputs, input_shape), **kwargs).shape
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
            getattr(JaxBackend, func)(*input_generator(num_inputs, input_shape)).shape
            == output_shape
        )


def test_concatenate():
    input = inputs(3) + inputs(2, (2, 1))

    # Without axis
    with pytest.raises(Exception):
        JaxBackend.concat(*input)

    # With axis
    assert JaxBackend.concat(*input, axis=1).shape == (2, 11)
    assert JaxBackend.concat(*inputs(1), axis=1).shape == (2, 3)


def test_stack():
    input = inputs(3) + inputs(2, (2, 1))

    assert JaxBackend.stack(*input, axis=0).shape == (5, 2, 3)
    assert JaxBackend.stack(*inputs(1), axis=0).shape == (1, 2, 3)
