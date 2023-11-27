from earthkit.data import FieldList


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
        assert ret.get("stepRange") is not None
        ret.setdefault("stepType", "max")
    else:
        step = int(step)
        ret["timeRangeIndicator"] = time_range_indicator(step)
    return ret


def extreme_grib_headers(clim: FieldList, ens: FieldList, num_steps: int) -> dict:
    extreme_headers = {}

    # EFI specific stuff
    ens_template = ens.metadata()[0].buffer_to_metadata()
    if int(ens_template.get("timeRangeIndicator")) == 3:
        if extreme_headers.get("numberIncludedInAverage") == 0:
            extreme_headers["numberIncludedInAverage"] = num_steps
        extreme_headers["numberMissingFromAveragesOrAccumulations"] = 0

    # set clim keys
    clim_template = clim.metadata()[0].buffer_to_metadata()
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


def threshold_grib_headers(edition: int, threshold_config: dict) -> dict:
    ret = {"paramId": threshold_config["out_paramid"]}
    if edition == 2:
        return ret

    print(threshold_config)
    threshold = threshold_config["value"]
    comparison = threshold_config["comparison"]
    if "localDecimalScaleFactor" in threshold_config:
        scale_factor = threshold_config["localDecimalScaleFactor"]
        ret["localDecimalScaleFactor"] = scale_factor
        threshold_value = round(threshold * 10**scale_factor, 0)
    else:
        threshold_value = threshold

    if "<" in comparison:
        ret.update({"thresholdIndicator": 2, "upperThreshold": threshold_value})
    elif ">" in comparison:
        ret.update({"thresholdIndicator": 1, "lowerThreshold": threshold_value})
    return ret
