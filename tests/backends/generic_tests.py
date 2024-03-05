import pytest


@pytest.mark.parametrize(
    "num_inputs, input_shape, kwargs, output_shape",
    [
        [4, (2, 3), {}, (2, 3)],
        [1, (2, 3), {}, ()],
    ],
)
def test_multi_arg(
    backend, input_generator, num_inputs, input_shape, kwargs, output_shape
):
    for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
        assert (
            getattr(backend, func)(
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
def test_two_arg(backend, input_generator, num_inputs, input_shape, output_shape):
    for func in ["add", "subtract", "multiply", "divide"]:
        assert (
            getattr(backend, func)(*input_generator(num_inputs, input_shape)).shape
            == output_shape
        )


@pytest.mark.parametrize(
    ["num_inputs", "shape"],
    [
        [3, (2, 3)],
        [1, (3,)],
    ],
)
def test_two_arg_raises(backend, input_generator, num_inputs, shape):
    with pytest.raises(Exception):
        backend.add(*input_generator(num_inputs, shape))


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[0], {"axis": 0}, (3,)],
        [[[0]], {"axis": 0}, (1, 3)],
        [[[0, 1]], {"axis": 1}, (2, 2)],
    ],
)
def test_take(backend, input_generator, args, kwargs, output_shape):
    output = backend.take(*input_generator(1), *args, **kwargs)
    assert output.shape == output_shape


def test_batchable(backend):
    for func in ["max", "min", "sum", "prod", "concat"]:
        assert getattr(backend, func).batchable
