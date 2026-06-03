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

from earthkit.workflows import backends


def _make_scalar_dicts(n: int, keys: tuple[str, ...] = ("a", "b", "c")) -> list[dict[str, float]]:
    """Generate n dicts with the given keys and sequential float values."""
    return [{k: float(i * len(keys) + j + 1) for j, k in enumerate(keys)} for i in range(n)]


def _make_xr_dicts(n: int, keys: tuple[str, ...] = ("a", "b"), shape: tuple[int, ...] = (2, 3)) -> list[dict[str, xr.DataArray]]:
    """Generate n dicts whose values are xarray DataArrays."""
    return [
        {
            k: xr.DataArray(
                np.random.rand(*shape) + i + j,
                dims=[f"dim{d}" for d in range(len(shape))],
            )
            for j, k in enumerate(keys)
        }
        for i in range(n)
    ]


class TestDictBackendAggregation:
    """Aggregation delegates to value-level backends."""

    @pytest.mark.parametrize("func", ["mean", "std", "max", "min", "sum", "prod", "var"])
    def test_single_dict_delegates_to_values(self, func: str) -> None:
        """Single dict: applies aggregation to each value independently."""
        d = _make_xr_dicts(1, keys=("x", "y"), shape=(4,))[0]
        result = getattr(backends, func)(d)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y"}
        # Each value should be reduced to a scalar
        for v in result.values():
            assert v.shape == ()

    @pytest.mark.parametrize("func", ["mean", "max", "min", "sum", "prod"])
    def test_multi_dict_aggregates_across_dicts(self, func: str) -> None:
        """Multiple dicts: aggregates corresponding values across dicts."""
        dicts = _make_xr_dicts(3, keys=("a", "b"), shape=(2, 3))
        result = getattr(backends, func)(*dicts)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        # Shape preserved (aggregation is across the dict axis, not spatial)
        for v in result.values():
            assert v.shape == (2, 3)

    @pytest.mark.parametrize("func", ["mean", "max", "min", "sum"])
    def test_aggregation_common_keys_only(self, func: str) -> None:
        """Only keys present in all dicts appear in the result."""
        a = {"x": xr.DataArray([1.0, 2.0]), "y": xr.DataArray([3.0, 4.0])}
        b = {"y": xr.DataArray([5.0, 6.0]), "z": xr.DataArray([7.0, 8.0])}
        result = getattr(backends, func)(a, b)
        assert set(result.keys()) == {"y"}

    def test_mean_values_correct(self) -> None:
        a = {"x": xr.DataArray([2.0, 4.0])}
        b = {"x": xr.DataArray([6.0, 8.0])}
        result = backends.mean(a, b)
        np.testing.assert_array_equal(result["x"].values, [4.0, 6.0])

    def test_sum_values_correct(self) -> None:
        a = {"x": xr.DataArray([1.0, 2.0])}
        b = {"x": xr.DataArray([3.0, 4.0])}
        result = backends.sum(a, b)
        np.testing.assert_array_equal(result["x"].values, [4.0, 6.0])

    def test_min_values_correct(self) -> None:
        a = {"x": xr.DataArray([1.0, 5.0])}
        b = {"x": xr.DataArray([3.0, 2.0])}
        result = backends.min(a, b)
        np.testing.assert_array_equal(result["x"].values, [1.0, 2.0])

    def test_max_values_correct(self) -> None:
        a = {"x": xr.DataArray([1.0, 5.0])}
        b = {"x": xr.DataArray([3.0, 2.0])}
        result = backends.max(a, b)
        np.testing.assert_array_equal(result["x"].values, [3.0, 5.0])


class TestDictBackendStack:
    def test_stack_merges_dicts(self) -> None:
        a = {"x": 1, "y": 2}
        b = {"z": 3, "w": 4}
        result = backends.stack(a, b)
        assert result == {"x": 1, "y": 2, "z": 3, "w": 4}

    def test_stack_later_overwrites(self) -> None:
        a = {"x": 1, "y": 2}
        b = {"y": 99, "z": 3}
        result = backends.stack(a, b)
        assert result == {"x": 1, "y": 99, "z": 3}

    def test_stack_single_dict(self) -> None:
        a = {"x": 1}
        result = backends.stack(a)
        assert result == {"x": 1}

    def test_stack_many_dicts(self) -> None:
        dicts = _make_scalar_dicts(5)
        result = backends.stack(*dicts)
        # All keys present, values from last dict with those keys win
        assert set(result.keys()) == {"a", "b", "c"}
        assert result == dicts[-1]

    def test_stack_empty_dicts(self) -> None:
        result = backends.stack({}, {}, {"a": 1})
        assert result == {"a": 1}


class TestDictBackendConcat:
    def test_concat_merges_dicts(self) -> None:
        a = {"x": 1}
        b = {"y": 2}
        result = backends.concat(a, b)
        assert result == {"x": 1, "y": 2}

    def test_concat_later_overwrites(self) -> None:
        a = {"x": 1, "y": 2}
        b = {"y": 99, "z": 3}
        result = backends.concat(a, b)
        assert result == {"x": 1, "y": 99, "z": 3}

    def test_concat_single_dict(self) -> None:
        a = {"x": 1, "y": 2}
        result = backends.concat(a)
        assert result == {"x": 1, "y": 2}

    def test_concat_many_dicts(self) -> None:
        a = {"a": 1}
        b = {"b": 2}
        c = {"c": 3}
        result = backends.concat(a, b, c)
        assert result == {"a": 1, "b": 2, "c": 3}


class TestDictBackendTake:
    def test_take_subset(self) -> None:
        d = {"a": 1, "b": 2, "c": 3}
        result = backends.take(d, ["a", "c"], dim=0)
        assert result == {"a": 1, "c": 3}

    def test_take_single_key(self) -> None:
        d = {"x": 10, "y": 20}
        result = backends.take(d, ["x"], dim=0)
        assert result == 10

    def test_take_all_keys(self) -> None:
        d = {"a": 1, "b": 2}
        result = backends.take(d, ["a", "b"], dim=0)
        assert result == {"a": 1, "b": 2}

    def test_take_preserves_order(self) -> None:
        d = {"a": 1, "b": 2, "c": 3}
        result = backends.take(d, ["c", "a"], dim=0)
        assert list(result.keys()) == ["c", "a"]

    def test_take_missing_key_raises(self) -> None:
        d = {"a": 1, "b": 2}
        with pytest.raises(KeyError, match="not found in array"):
            backends.take(d, ["a", "missing"], dim=0)

    def test_take_all_missing_raises(self) -> None:
        d = {"a": 1}
        with pytest.raises(KeyError):
            backends.take(d, ["x", "y"], dim=0)


class TestDictBackendArithmetic:
    """Arithmetic delegates to value-level backends."""

    def test_add_scalars(self) -> None:
        a = {"x": 1, "y": 2}
        b = {"x": 10, "y": 20}
        result = backends.add(a, b)
        assert result == {"x": 11, "y": 22}

    def test_subtract_scalars(self) -> None:
        a = {"x": 10, "y": 20}
        b = {"x": 3, "y": 5}
        result = backends.subtract(a, b)
        assert result == {"x": 7, "y": 15}

    def test_multiply_scalars(self) -> None:
        a = {"x": 3, "y": 4}
        b = {"x": 2, "y": 5}
        result = backends.multiply(a, b)
        assert result == {"x": 6, "y": 20}

    def test_divide_scalars(self) -> None:
        a = {"x": 10.0, "y": 20.0}
        b = {"x": 2.0, "y": 5.0}
        result = backends.divide(a, b)
        assert result == {"x": 5.0, "y": 4.0}

    def test_pow_scalars(self) -> None:
        a = {"x": 2, "y": 3}
        b = {"x": 3, "y": 2}
        result = backends.pow(a, b)
        assert result == {"x": 8, "y": 9}

    def test_add_xarray_values(self) -> None:
        """Arithmetic delegates to xarray backend when values are DataArrays."""
        a = {"x": xr.DataArray([1.0, 2.0]), "y": xr.DataArray([3.0, 4.0])}
        b = {"x": xr.DataArray([10.0, 20.0]), "y": xr.DataArray([30.0, 40.0])}
        result = backends.add(a, b)
        assert set(result.keys()) == {"x", "y"}
        np.testing.assert_array_equal(result["x"].values, [11.0, 22.0])
        np.testing.assert_array_equal(result["y"].values, [33.0, 44.0])

    def test_arithmetic_common_keys_only(self) -> None:
        """When dicts have different keys, only common keys appear in result."""
        a = {"x": 1, "y": 2, "z": 3}
        b = {"y": 10, "z": 20, "w": 30}
        result = backends.add(a, b)
        assert result == {"y": 12, "z": 23}

    def test_arithmetic_no_common_keys(self) -> None:
        a = {"x": 1}
        b = {"y": 2}
        result = backends.add(a, b)
        assert result == {}

    def test_divide_by_zero_raises(self) -> None:
        a = {"x": 1}
        b = {"x": 0}
        with pytest.raises(ZeroDivisionError):
            backends.divide(a, b)

    @pytest.mark.parametrize("func", ["add", "subtract", "multiply", "divide", "pow"])
    def test_two_arg_enforced(self, func: str) -> None:
        """Arithmetic ops require exactly two arguments."""
        dicts = _make_scalar_dicts(3)
        with pytest.raises(Exception):
            getattr(backends, func)(*dicts)
