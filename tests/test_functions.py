import pytest
import numpy.random as random
import numpy as np

from earthkit.data import FieldList
from earthkit.data.core.metadata import RawMetadata

from cascade import functions


class MockMetaData(RawMetadata):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def buffer_to_metadata(self) -> "MockMetaData":
        return self


def random_fieldlist(*shape) -> FieldList:
    return FieldList.from_numpy(
        random.rand(*shape), [MockMetaData() for x in range(shape[0])]
    )


@pytest.mark.parametrize(
    "func",
    [
        functions.mean,
        functions.std,
        functions.maximum,
        functions.minimum,
    ],
)
def test_multi_arg(func):
    arr = [random_fieldlist(1, 5) for _ in range(5)]
    unnested = func(*arr)
    nested = func(functions.concatenate(*arr))
    assert np.all(unnested.values == nested.values)


@pytest.mark.parametrize(
    "func",
    [
        functions.add,
        functions.subtract,
        functions.multiply,
        functions.divide,
        functions.norm,
    ],
)
def test_two_arg(func):
    arr = [random_fieldlist(1, 5) for _ in range(2)]
    unnested = func(*arr)
    nested = func(functions.concatenate(*arr))
    assert np.all(unnested.values == nested.values)


@pytest.mark.parametrize(
    "func",
    [
        functions.divide,
        functions.norm,
    ],
)
def test_two_arg(func):
    arr = [random_fieldlist(1, 5) for _ in range(3)]
    with pytest.raises(AssertionError):
        func(*arr)
    with pytest.raises(AssertionError):
        func(functions.concatenate(*arr))


@pytest.mark.parametrize(
    "comparison",
    [
        "<=",
        "<",
        ">=",
        ">",
    ],
)
def test_threshold(comparison):
    config = {"comparison": comparison, "value": 2, "out_paramid": 120}
    functions.threshold(config, random_fieldlist(1, 5))


def test_extreme():
    ens = random_fieldlist(5, 5)
    ens.metadata()[0]._d.update(
        {
            "timeRangeIndicator": 3,
            "date": "0",
            "subCentre": 0,
            "totalNumber": 0,
        }
    )
    clim = random_fieldlist(101, 5)
    clim.metadata()[0]._d.update(
        {
            "powerOfTenUsedToScaleClimateWeight": 0,
            "weightAppliedToClimateMonth1": 0,
            "firstMonthUsedToBuildClimateMonth1": 0,
            "lastMonthUsedToBuildClimateMonth1": 0,
            "firstMonthUsedToBuildClimateMonth2": 0,
            "lastMonthUsedToBuildClimateMonth2": 0,
            "numberOfBitsContainingEachPackedValue": 0,
        }
    )
    functions.efi(clim, ens, 0.0001, 2)
    functions.sot(clim, ens, 90, 0.0001, 2)


def test_quantiles():
    ens = random_fieldlist(5, 5)
    functions.quantiles(ens, 0.1)


def test_filter():
    arr = [random_fieldlist(1, 5) for _ in range(2)]
    functions.filter("<", 2, arr[0], arr[1], 0)
