import pytest

from cascade import backends


@pytest.mark.parametrize(
    "num_inputs, input_shape, kwargs, output_shape",
    [
        [4, (2, 3), {}, (2, 3)],
        [1, (2, 3), {}, ()],
    ],
)
def test_multi_arg(input_generator, num_inputs, input_shape, kwargs, output_shape):
    for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
        assert (
            getattr(backends, func)(
                *input_generator(num_inputs, input_shape), **kwargs
            ).shape
            == output_shape
        )


@pytest.mark.parametrize(
    ["num_inputs", "input_shape", "output_shape"],
    [
        [2, (2, 3), (2, 3)],
    ],
)
def test_two_arg(input_generator, num_inputs, input_shape, output_shape):
    for func in ["add", "subtract", "multiply", "divide", "pow"]:
        assert (
            getattr(backends, func)(*input_generator(num_inputs, input_shape)).shape
            == output_shape
        )


@pytest.mark.parametrize(
    ["num_inputs", "shape"],
    [
        [3, (2, 3)],
        [1, (3,)],
    ],
)
def test_two_arg_raises(input_generator, num_inputs, shape):
    with pytest.raises(Exception):
        backends.add(*input_generator(num_inputs, shape))


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[0], {"axis": 0}, (3,)],
        [[[0]], {"axis": 0}, (1, 3)],
        [[[0, 1]], {"axis": 1}, (2, 2)],
    ],
)
def test_take(input_generator, args, kwargs, output_shape):
    output = backends.take(*input_generator(1), *args, **kwargs)
    assert output.shape == output_shape


def test_batchable():
    for func in ["max", "min", "sum", "prod", "concat"]:
        assert getattr(backends, func).batchable
