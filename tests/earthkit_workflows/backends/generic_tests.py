# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from earthkit.workflows import backends


class BackendBase:
    def input_generator(self, *args):
        raise NotImplementedError

    def shape(self, array):
        raise NotImplementedError

    @pytest.mark.parametrize(
        "num_inputs, input_shape, kwargs, output_shape",
        [
            [4, (2, 3), {}, (2, 3)],
            [1, (2, 3), {}, ()],
        ],
    )
    def test_multi_arg(self, num_inputs, input_shape, kwargs, output_shape):
        for func in ["mean", "std", "max", "min", "sum", "prod", "var"]:
            assert (
                self.shape(
                    getattr(backends, func)(
                        *self.input_generator(num_inputs, input_shape), **kwargs
                    )
                )
                == output_shape
            )

    @pytest.mark.parametrize(
        ["num_inputs", "input_shape", "output_shape"],
        [
            [2, (2, 3), (2, 3)],
        ],
    )
    def test_two_arg(self, num_inputs, input_shape, output_shape):
        for func in ["add", "subtract", "multiply", "divide", "pow"]:
            assert (
                self.shape(
                    getattr(backends, func)(
                        *self.input_generator(num_inputs, input_shape)
                    )
                )
                == output_shape
            )

    @pytest.mark.parametrize(
        ["num_inputs", "shape"],
        [
            [3, (2, 3)],
            [1, (3,)],
        ],
    )
    def test_two_arg_raises(self, num_inputs, shape):
        with pytest.raises(Exception):
            backends.add(*self.input_generator(num_inputs, shape))

    @pytest.mark.parametrize(
        ["args", "kwargs", "output_shape"],
        [
            [[0], {"dim": 0}, (3,)],
            [[[0]], {"dim": 0}, (1, 3)],
            [[[0, 1]], {"dim": 1}, (2, 2)],
        ],
    )
    def test_take(self, args, kwargs, output_shape):
        output = backends.take(*self.input_generator(1), *args, **kwargs)
        assert self.shape(output) == output_shape

    def test_batchable(self):
        for func in ["max", "min", "sum", "prod", "concat"]:
            assert getattr(backends, func).batchable
