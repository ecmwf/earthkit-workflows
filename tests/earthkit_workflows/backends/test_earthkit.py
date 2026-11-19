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
from earthkit.data import Field, FieldList

from earthkit.workflows import backends
from earthkit.workflows.backends.earthkit import FieldListBackend


def random_fieldlist(*shape: int) -> FieldList:
    return FieldList.from_fields([Field.from_components(values=row) for row in random.rand(*shape)])


def fieldlist_from_values(values: np.ndarray, labels: dict[str, list] | None = None) -> FieldList:
    """One field per row of values, with labels[name][i] as the label `name` of field i"""
    fields = []
    for i, row in enumerate(values):
        field_labels = {name: column[i] for name, column in (labels or {}).items()}
        fields.append(Field.from_components(values=row, labels=field_labels) if field_labels else Field.from_components(values=row))
    return FieldList.from_fields(fields)


def to_array(fl: FieldList) -> np.ndarray:
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
        "mean",
        "std",
        "max",
        "min",
        "prod",
        "sum",
        "norm",
    ],
)
def test_multi_arg(func, input_generator, values):
    arr = [input_generator(2, 20), input_generator(2, 20), input_generator(1, 20)]
    concat = backends.method("concat", *arr)
    assert len(concat) == 5
    assert values(concat).shape == (5, 20)
    nested = backends.method(func, concat, backend_kwargs={"metadata": {"labels.key": "value"}})
    assert nested.get("labels.key") == ["value"]
    assert values(nested).shape == (1, 20)

    with pytest.raises(ValueError):
        backends.method(func, *arr)

    # Field lists with multiple fields
    arr2 = [input_generator(3, 20) for _ in range(5)]
    assert values(backends.method(func, *arr2)).shape == (3, 20)


@pytest.mark.parametrize(
    "func",
    [
        "add",
        "subtract",
        "multiply",
        "divide",
        "pow",
        "diff",
    ],
)
def test_two_arg(input_generator, values, func):
    # Single field in each field list
    arr = [input_generator(1, 5) for _ in range(2)]
    assert values(backends.method(func, *arr)).shape == (1, 5)

    # Multiple fields in each field list
    arr = [input_generator(3, 5), 2]
    unnested = backends.method(func, *arr)
    assert values(unnested).shape == (3, 5)

    # Raises on too many arguments
    arr = [input_generator(1, 5) for _ in range(3)]
    with pytest.raises(TypeError, match="takes 3 positional arguments but 4 were given"):
        backends.method(func, *arr)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'arr2'"):
        backends.method(func, backends.method("concat", *arr))


def test_divide_and_pow_values():
    a = fieldlist_from_values(np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]))
    b = fieldlist_from_values(np.array([[2.0, 2.0, 3.0], [1.0, 4.0, 2.0]]))
    np.testing.assert_allclose(backends.method("divide", a, b).values, a.values / b.values)
    np.testing.assert_allclose(backends.method("pow", a, b).values, a.values**b.values)
    np.testing.assert_allclose(backends.method("divide", a, 2).values, a.values / 2)
    np.testing.assert_allclose(backends.method("pow", a, 2).values, a.values**2)
    np.testing.assert_allclose(backends.method("diff", a, b).values, b.values - a.values)


def test_concat(input_generator, values):
    arr = [input_generator(1, 5) for _ in range(3)]
    res = backends.method("concat", *arr)
    assert len(res) == 3
    assert np.all(values(res[0]) == values(arr[0]))
    assert np.all(values(res[-1]) == values(arr[-1]))


def test_concat_keeps_values_and_count(input_generator, values):
    arr = [input_generator(2, 5), input_generator(3, 5), input_generator(1, 5)]
    res = backends.method("concat", *arr)
    assert len(res) == 6
    np.testing.assert_array_equal(values(res), np.vstack([values(x) for x in arr]))
    same = [input_generator(2, 5), input_generator(2, 5)]
    res = backends.method("concat", *same)
    assert len(res) == 4
    np.testing.assert_array_equal(values(res), np.vstack([values(x) for x in same]))


def test_stack(input_generator, values):
    arr = [input_generator(1, 5) for _ in range(3)]
    res = backends.method("stack", *arr)
    assert len(res) == 3
    np.testing.assert_array_equal(values(res), np.vstack([values(x) for x in arr]))
    with pytest.raises(TypeError, match="does not accept keyword arguments"):
        backends.method("stack", *arr, backend_kwargs={"a": 1})
    with pytest.raises(AssertionError, match="Can not stack"):
        backends.method("stack", input_generator(2, 5), input_generator(1, 5))


def test_filter(values):
    arr = fieldlist_from_values(np.arange(8.0).reshape(2, 4), {"k": ["a", "b"]})
    mask = fieldlist_from_values(np.array([[1, 0, 0, 1], [0, 0, 1, 0]]))
    res = backends.method("filter", arr, mask, -1.0)
    np.testing.assert_array_equal(values(res), np.array([[-1.0, 1.0, 2.0, -1.0], [4.0, 5.0, -1.0, 7.0]]))
    assert res.get("labels.k") == ["a", "b"]
    res = backends.method("filter", arr, mask, backend_kwargs={"metadata": {"labels.x": 1}})
    assert res.get("labels.x") == [1, 1]
    assert values(res)[0, 0] == 0.0


@pytest.mark.parametrize(
    ["args", "kwargs", "output_shape"],
    [
        [[0], {"dim": 0}, (1, 6)],
        [[[0]], {"dim": 0}, (1, 6)],
        [[[0, 1]], {"dim": 0}, (2, 6)],
        [[slice(1, 3)], {"dim": 0}, (2, 6)],
    ],
)
def test_take(input_generator, values, args, kwargs, output_shape):
    input = input_generator(3, 6)
    output = backends.method("take", input, *args, **kwargs)
    assert values(output).shape == output_shape


def test_take_slice_values():
    fl = fieldlist_from_values(np.arange(12.0).reshape(4, 3))
    np.testing.assert_array_equal(backends.method("take", fl, 2).values, fl.values[[2]])
    np.testing.assert_array_equal(backends.method("take", fl, [3, 0]).values, fl.values[[3, 0]])
    np.testing.assert_array_equal(backends.method("take", fl, slice(1, 3)).values, fl.values[1:3])
    with pytest.raises(ValueError, match="dim != 0"):
        backends.method("take", fl, 0, dim=1)
    with pytest.raises(ValueError, match="empty fieldlist"):
        backends.method("take", fl, [], dim=0)


@pytest.fixture
def labelled() -> FieldList:
    return fieldlist_from_values(
        np.arange(24.0).reshape(6, 4),
        {"k": [3, 1, 2, 1, 3, 2], "other": ["a", "b", "c", "d", "e", "f"]},
    )


def test_take_sel(labelled):
    res = backends.method("take", labelled, 1, dim="labels.k", backend_kwargs={"method": "sel"})
    assert isinstance(res, FieldList)
    assert res.get("labels.k") == [1, 1]
    np.testing.assert_array_equal(res.values, labelled.values[[1, 3]])
    res = backends.method("take", labelled, [1, 3], dim="labels.k", backend_kwargs={"method": "sel"})
    assert sorted(res.get("labels.k")) == [1, 1, 3, 3]
    np.testing.assert_array_equal(res.values, labelled.values[[0, 1, 3, 4]])
    res = backends.method("take", labelled, [1, 3], dim="labels.k", backend_kwargs={"method": "sel", "labels.other": "a"})
    assert res.get("labels.other") == ["a"]


@pytest.mark.parametrize(
    ["indices", "expected_k"],
    [
        [0, [1, 1]],
        [-1, [3, 3]],
        [[0, 2], [3, 1, 3, 1]],
        [slice(1, None), [3, 2, 3, 2]],
        [slice(None, None, 2), [3, 1, 3, 1]],
    ],
)
def test_take_isel(labelled, indices, expected_k):
    res = backends.method("take", labelled, indices, dim="labels.k", backend_kwargs={"method": "isel"})
    assert isinstance(res, FieldList)
    assert sorted(res.get("labels.k")) == sorted(expected_k)
    positions = [i for i, k in enumerate(labelled.get("labels.k")) if k in set(expected_k)]
    np.testing.assert_array_equal(res.values, labelled.values[positions])


@pytest.mark.parametrize("indices", [3, -4, [0, 3]])
def test_take_isel_out_of_range(labelled, indices):
    with pytest.raises(ValueError, match="out of range"):
        backends.method("take", labelled, indices, dim="labels.k", backend_kwargs={"method": "isel"})


def test_take_isel_empty_slice(labelled):
    with pytest.raises(ValueError, match="empty fieldlist"):
        backends.method("take", labelled, slice(5, 7), dim="labels.k", backend_kwargs={"method": "isel"})


def test_take_errors(labelled):
    with pytest.raises(ValueError, match="dim must be a string"):
        backends.method("take", labelled, 0, backend_kwargs={"method": "sel"})
    with pytest.raises(ValueError, match="Invalid method"):
        backends.method("take", labelled, 0, dim="labels.k", backend_kwargs={"method": "foo"})
    with pytest.raises(ValueError, match="resulted in empty fieldlist: indices"):
        backends.method("take", labelled, 1, dim="k", backend_kwargs={"method": "sel"})


def test_set_metadata(labelled, values):
    res = backends.method("set_metadata", labelled, {"labels.name": "x"})
    assert res.get("labels.name") == ["x"] * 6
    assert res.get("labels.k") == labelled.get("labels.k")
    np.testing.assert_array_equal(values(res), values(labelled))


def test_set_metadata_invalid_key(labelled):
    with pytest.raises(ValueError, match="key") as exc_info:
        backends.method("set_metadata", labelled, {"key": "value"})
    assert isinstance(exc_info.value.__cause__, KeyError)
    with pytest.raises(ValueError, match="key"):
        backends.method("mean", labelled, backend_kwargs={"metadata": {"key": "value"}})


def test_callable_metadata():
    fl = fieldlist_from_values(np.arange(8.0).reshape(2, 4))
    res = backends.method("mean", fl, backend_kwargs={"metadata": lambda *arrays: {"labels.n": len(arrays)}})
    assert res.get("labels.n") == [1]


def test_array_module():
    fl = random_fieldlist(3, 4)
    assert backends.array_module(fl) is FieldListBackend
    assert backends.array_module(backends.method("concat", fl, fl)) is FieldListBackend
    selected = fieldlist_from_values(np.arange(8.0).reshape(2, 4), {"k": [1, 2]}).sel(**{"labels.k": 1})
    assert backends.array_module(selected) is FieldListBackend
