# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import numpy.random as random
import pytest
from earthkit.data import FieldList, SimpleFieldList
from earthkit.data.core.metadata import RawMetadata

from earthkit.workflows import backends


class MockMetaData(RawMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def random_fieldlist(*shape) -> SimpleFieldList:
    return FieldList.from_array(
        random.rand(*shape), [MockMetaData() for x in range(shape[0])]
    )


def to_array(fl: SimpleFieldList):
    return fl.values


@pytest.fixture
def input_generator():
    return random_fieldlist


@pytest.fixture
def values():
    return to_array


@pytest.mark.parametrize(
    "func",
    [
        backends.mean,
        backends.std,
        backends.max,
        backends.min,
        backends.prod,
        backends.sum,
        backends.norm,
    ],
)
def test_multi_arg(func, input_generator, values):
    arr = [input_generator(2, 20), input_generator(2, 20), input_generator(1, 20)]
    concat = backends.concat(*arr)
    assert len(concat) == 5
    assert values(concat).shape == (5, 20)
    nested = func(concat)
    assert values(nested).shape == (1, 20)

    with pytest.raises(ValueError):
        func(*arr)

    # Field lists with multiple fields
    arr2 = [input_generator(3, 20) for _ in range(5)]
    assert values(func(*arr2)).shape == (3, 20)


@pytest.mark.parametrize(
    "func",
    [
        backends.add,
        backends.subtract,
        backends.multiply,
        backends.divide,
        backends.diff,
    ],
)
def test_two_arg(input_generator, values, func):
    # Single field in each field list
    arr = [input_generator(1, 5) for _ in range(2)]
    assert values(func(*arr)).shape == (1, 5)

    # Multiple fields in each field list
    arr = [input_generator(3, 5), 2]
    unnested = func(*arr)
    assert values(unnested).shape == (3, 5)

    # Raises on too many arguments
    arr = [input_generator(1, 5) for _ in range(3)]
    with pytest.raises(AssertionError):
        func(*arr)
    with pytest.raises(AssertionError):
        func(backends.concat(*arr))


def test_concat(input_generator, values):
    arr = [input_generator(1, 5) for _ in range(3)]
    res = backends.concat(*arr)
    assert len(res) == 3
    assert np.all(values(res[0]) == values(arr[0]))
    assert np.all(values(res[-1]) == values(arr[-1]))


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[0], {"dim": 0}, (1, 6)],
        [[[0]], {"dim": 0}, (1, 6)],
        [[[0, 1]], {"dim": 0}, (2, 6)],
    ],
)
def test_take(input_generator, values, args, kwargs, output_shape):
    input = input_generator(3, 6)
    output = backends.take(input, *args, **kwargs)
    assert values(output).shape == output_shape
