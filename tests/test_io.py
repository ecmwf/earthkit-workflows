import pytest

from cascade.io import retrieve, write


def test_retrieve(tmpdir):
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
    data = retrieve("mars", request)
    write(f"{tmpdir}/test.grib", data, {})


def test_retrieve_fail():
    with pytest.raises(Exception):
        retrieve(
            "fdb",
            {
                "class": "od",
                "expver": "0001",
                "stream": "enfo",
                "type": "cf",
                "date": "20230101",
                "time": "12",
                "domain": "g",
                "levtype": "sfc",
                "step": "12",
                "param": 228,
            },
        )
