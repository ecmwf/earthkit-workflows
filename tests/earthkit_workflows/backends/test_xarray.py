# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import xarray as xr
from generic_tests import BackendBase

from earthkit.workflows import backends


class XarrayBackend(BackendBase):
    @pytest.mark.parametrize(
        ["num_inputs", "kwargs", "output_shape"],
        [
            [1, {"dim": "dim0"}, (3,)],
            [1, {"dim": "dim1"}, (2,)],
        ],
    )
    def test_multi_arg_dim(self, num_inputs, kwargs, output_shape):
        for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
            assert self.shape(
                (getattr(backends, func)(*self.input_generator(num_inputs), **kwargs))
                == output_shape
            )

    def test_concatenate(self):
        input = self.input_generator(3) + self.input_generator(2, (2, 1))

        # Without dim
        with pytest.raises(TypeError):
            backends.concat(*input)

        # With dim
        assert self.shape(backends.concat(*input, dim="dim1")) == (2, 11)
        assert self.shape(backends.concat(*self.input_generator(1), dim="dim1")) == (
            2,
            3,
        )

    def test_stack(self):
        input = self.input_generator(3) + self.input_generator(2, (2,))

        x = backends.stack(*input, dim="NEW", coords="minimal")
        assert self.shape(x) == (5, 2, 3)
        assert not np.any(np.isnan(self.values(x)))

        # Without dim
        with pytest.raises(TypeError):
            backends.stack(*input)

        # With existing dim
        with pytest.raises(ValueError):
            backends.stack(*input, dim="dim0")

        # With dim and axis
        y = backends.stack(*input, axis=2, dim="NEW", coords="minimal")
        assert np.all(x.transpose("dim0", "dim1", "NEW") == y)
        assert self.shape(
            backends.stack(*self.input_generator(1), axis=0, dim="NEW")
        ) == (
            1,
            2,
            3,
        )

    @pytest.mark.parametrize(
        ["args", "kwargs", "output_shape"],
        [
            [[0], {"dim": "dim0"}, (3,)],
            [[[0]], {"dim": "dim0"}, (1, 3)],
            [[[0, 1]], {"dim": "dim1"}, (2, 2)],
            [[[0, 1]], {"dim": 1}, (2, 2)],
            [[[0, 1]], {"dim": "dim1", "method": "sel"}, (2, 2)],
        ],
    )
    def test_take_extended(self, args, kwargs, output_shape):
        output = backends.take(*self.input_generator(1), *args, **kwargs)
        assert self.shape(output) == output_shape


class TestXarrayDataArrayBackend(XarrayBackend):
    def input_generator(self, number: int, shape=(2, 3)):
        return [
            xr.DataArray(
                np.random.rand(*shape),
                dims=[f"dim{x}" for x in range(len(shape))],
                coords={f"dim{x}": range(shape[x]) for x in range(len(shape))},
            )
            for _ in range(number)
        ]

    def shape(self, array):
        return array.shape

    def values(self, array):
        return array.values


class TestXarrayDatasetBackend(XarrayBackend):
    def input_generator(self, number: int, shape=(2, 3)):
        return [
            xr.Dataset(
                {
                    "test": xr.DataArray(
                        np.random.rand(*shape),
                        dims=[f"dim{x}" for x in range(len(shape))],
                        coords={f"dim{x}": range(shape[x]) for x in range(len(shape))},
                    )
                }
            )
            for _ in range(number)
        ]

    def shape(self, array):
        return array["test"].shape

    def values(self, array):
        return array["test"].values
