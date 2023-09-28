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
        ret.setdefault("max")
    else:
        step = int(step)
        ret["timeRangeIndicator"] = time_range_indicator(step)
    return ret
