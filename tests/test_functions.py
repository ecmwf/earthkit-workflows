import pytest
import numpy.random as random
import importlib

from earthkit.data import FieldList

from cascade.grib import GribBufferMetaData
from cascade import grib
from cascade import functions


# Monkey patch extracting grib template from buffer
def buffer_to_template(metadata):
    return metadata


def random_fieldlist(*shape) -> FieldList:
    return FieldList.from_numpy(
        random.rand(*shape), [GribBufferMetaData(None) for x in range(shape[0])]
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
    func(*arr)


@pytest.mark.parametrize(
    "func",
    [
        functions.add,
        functions.subtract,
        functions.multiply,
        functions.divide,
    ],
)
def test_two_arg(monkeypatch, func):
    arr = [random_fieldlist(1, 5) for _ in range(2)]
    monkeypatch.setattr(functions, "buffer_to_template", buffer_to_template)
    func(*arr)


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


def test_extreme(monkeypatch):
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
    monkeypatch.setattr(grib, "buffer_to_template", buffer_to_template)
    importlib.reload(functions)
    functions.efi(clim, ens, 0.0001, 2)
    functions.sot(clim, ens, 90, 0.0001, 2)


def test_quantiles():
    ens = random_fieldlist(5, 5)
    functions.quantiles(ens, 0.1)


def test_wind_speed():
    # Wind speed computed with u/v params inside a single field
    functions.wind_speed(random_fieldlist(2, 5))

    # Error when number of fields in list is not equal to 2
    with pytest.raises(AssertionError):
        functions.wind_speed(functions.wind_speed(random_fieldlist(6, 5)))


def test_filter():
    arr = [random_fieldlist(1, 5) for _ in range(2)]
    functions.filter("<", 2, arr[0], arr[1], 0)
