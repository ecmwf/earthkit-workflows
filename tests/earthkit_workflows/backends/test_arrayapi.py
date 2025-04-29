# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
from generic_tests import BackendBase

from earthkit.workflows import backends


class TestArrayAPIBackend(BackendBase):
    def input_generator(self, num_inputs: int, input_shape=(2, 3)):
        return [np.random.rand(*input_shape) for _ in range(num_inputs)]

    def shape(self, array):
        return array.shape

    @pytest.mark.parametrize(
        ["num_inputs", "input_shape", "kwargs", "output_shape"],
        [
            [1, (2, 3), {"axis": 0}, (3,)],
            [1, (2, 3), {"axis": 1}, (2,)],
        ],
    )
    def test_multi_arg_axis(self, num_inputs, input_shape, kwargs, output_shape):
        for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
            assert (
                getattr(backends, func)(
                    *self.input_generator(num_inputs, input_shape), **kwargs
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
    def test_two_arg_single(self, num_inputs, input_shape, output_shape):
        for func in ["add", "subtract", "multiply", "divide"]:
            assert (
                getattr(backends, func)(
                    *self.input_generator(num_inputs, input_shape)
                ).shape
                == output_shape
            )

    def test_concatenate(self):
        input = self.input_generator(3) + self.input_generator(2, (2, 1))

        # Without axis
        with pytest.raises(Exception):
            backends.concat(*input)

        # With axis
        assert backends.concat(*input, axis=1).shape == (2, 11)
        assert backends.concat(*self.input_generator(1), axis=1).shape == (2, 3)

    def test_stack(self):
        input = self.input_generator(3) + self.input_generator(2, (2, 1))

        assert backends.stack(*input, axis=0).shape == (5, 2, 3)
        assert backends.stack(*self.input_generator(1), axis=0).shape == (1, 2, 3)
