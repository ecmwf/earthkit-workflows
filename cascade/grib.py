from earthkit.data.readers.grib.metadata import GribMetadata
from earthkit.data.readers.grib.memory import GribMessageMemoryReader
from earthkit.data.readers.grib.codes import GribCodesHandle


def buffer_to_template(buffer):
    return GribMetadata(
        GribCodesHandle(GribMessageMemoryReader(buffer)._next_handle(), None, None)
    )


def time_range_indicator(step: int) -> int:
    if step == 0:
        return 1
    if step > 255:
        return 10
    return 0


def basic_headers(grib_sets: dict) -> dict:
    ret = grib_sets.copy()
    step = ret.get("step")
    if step is None:
        step_range = ret.get("stepRange")
        assert step_range is not None
        ret.setdefault("stepType", "max")
    else:
        step = int(step)
        ret["timeRangeIndicator"] = time_range_indicator(step)
    return ret


def extreme_grib_headers(clim, ens):
    extreme_headers = {}

    # EFI specific stuff
    ens_template = buffer_to_template(ens.metadata()[0].get("buffer"))
    if int(ens_template.get("timeRangeIndicator")) == 3:
        if extreme_headers.get("numberIncludedInAverage") == 0:
            extreme_headers["numberIncludedInAverage"] = len(window.steps)
        extreme_headers["numberMissingFromAveragesOrAccumulations"] = 0

    # set clim keys
    clim_template = buffer_to_template(clim.metadata()[0].get("buffer"))
    clim_keys = [
        "powerOfTenUsedToScaleClimateWeight",
        "weightAppliedToClimateMonth1",
        "firstMonthUsedToBuildClimateMonth1",
        "lastMonthUsedToBuildClimateMonth1",
        "firstMonthUsedToBuildClimateMonth2",
        "lastMonthUsedToBuildClimateMonth2",
        "numberOfBitsContainingEachPackedValue",
    ]
    for key in clim_keys:
        extreme_headers[key] = clim_template.get(key)

    fc_keys = [
        "date",
        "subCentre",
        "totalNumber",
    ]
    for key in fc_keys:
        extreme_headers[key] = ens_template.get(key)

    extreme_headers["ensembleSize"] = len(ens.values)

    return extreme_headers
