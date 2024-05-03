import numpy as np
import pytest
import xarray as xr
from generic_tests import BackendBase

from cascade import backends


class TestXarrayBackend(BackendBase):
    def input_generator(self, number: int, shape=(2, 3)):
        return [
            xr.DataArray(
                np.random.rand(*shape), dims=[f"dim{x}" for x in range(len(shape))]
            )
            for _ in range(number)
        ]

    @pytest.mark.parametrize(
        ["num_inputs", "kwargs", "output_shape"],
        [
            [1, {"dim": "dim0"}, (3,)],
            [1, {"dim": "dim1"}, (2,)],
        ],
    )
    def test_multi_arg_dim(self, num_inputs, kwargs, output_shape):
        for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
            assert (
                getattr(backends, func)(
                    *self.input_generator(num_inputs), **kwargs
                ).shape
                == output_shape
            )

    def test_concatenate(self):
        # Note broadcasting in xarray works different to numpy,
        # where an array with shape (2, 1) can not be boradcasted to (2, 3)
        # whereas an array with a missing dimension e.g. (2,) can be broadcasted to
        # shape (2, 3)
        input = self.input_generator(3) + self.input_generator(2, (2,))

        # Without dim
        with pytest.raises(Exception):
            backends.concat(*input)

        # With dim
        assert backends.concat(*input, dim="dim1").shape == (2, 11)
        assert backends.concat(*self.input_generator(1), dim="dim1").shape == (2, 3)

    def test_stack(self):
        input = self.input_generator(3) + self.input_generator(2, (2,))

        x = backends.stack(*input, dim="NEW")
        assert x.shape == (5, 2, 3)
        assert not np.any(np.isnan(x))

        # Without dim
        with pytest.raises(Exception):
            backends.stack(*input)

        # With existing dim
        with pytest.raises(Exception):
            backends.stack(*input, dim="dim0")

        # With dim and axis
        y = backends.stack(*input, axis=2, dim="NEW")
        assert np.all(x.transpose("dim0", "dim1", "NEW") == y)
        assert backends.stack(*self.input_generator(1), axis=0, dim="NEW").shape == (
            1,
            2,
            3,
        )
