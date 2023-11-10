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
