from datetime import datetime, timedelta
import numpy as np

from cascade.io import retrieve, write

request = {
    "class": "od",
    "expver": "0001",
    "stream": "enfo",
    "type": "cf",
    "date": (datetime.today() - timedelta(days=1)).strftime("%Y%m%d"),
    "time": "12",
    "domain": "g",
    "levtype": "sfc",
    "step": "12",
    "param": 228,
    "source": "mars",
}


def test_retrieve(tmpdir):
    # Retrieve from single source
    data = retrieve(request)
    write(f"{tmpdir}/test.grib", data, {"step": 12})

    # Retrieve with multiple sources
    fdb_request = request.copy()
    fdb_request["source"] = "fdb"
    data2 = retrieve([request, fdb_request])
    assert np.all(data.values == data2.values)
