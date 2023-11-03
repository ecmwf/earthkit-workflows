import pytest
from datetime import datetime, timedelta
import dill
import multiprocessing

from cascade.io import retrieve, write

dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump
multiprocessing.queues._ForkingPickler = dill.Pickler
import concurrent.futures as fut

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
}


def test_retrieve(tmpdir):
    data = retrieve("mars", request)
    write(f"{tmpdir}/test.grib", data, {"step": 12})


def test_retrieve_fail():
    new_request = request.copy()
    new_request["date"] = "20230101"
    with pytest.raises(Exception):
        retrieve(
            "fdb",
            new_request,
        )


def test_multiprocess(tmpdir):
    futures = []
    base_request = request.copy()
    with fut.ProcessPoolExecutor(max_workers=2) as executor:
        for x in range(1, 2):
            base_request["type"] = "pf"
            base_request["number"] = x
            futures.append(executor.submit(retrieve, "mars", base_request))

    for future in fut.as_completed(futures):
        data = future.result()
        write(f"{tmpdir}/test.grib", data, {"step": 12})
