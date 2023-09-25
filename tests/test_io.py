import pytest

from cascade.io import retrieve, write


@pytest.mark.parametrize("source", ["fdb", "mars"])
def test_retrieve(tmpdir, source):
    request = {
        "class": "od",
        "expver": "0001",
        "stream": "enfo",
        "type": "cf",
        "date": "20230914",
        "time": "12",
        "domain": "g",
        "levtype": "sfc",
        "step": "12",
        "param": 228,
    }
    data = retrieve(source, request)
    write(f"{tmpdir}/test.grib", data, {})
