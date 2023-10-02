import pytest

from earthkit.data import FieldList

from cascade import functions
from cascade.io import retrieve

request = {
    "class": "od",
    "expver": "0001",
    "stream": "enfo",
    "type": "pf",
    "date": "20230914",
    "time": "12",
    "domain": "g",
    "levtype": "sfc",
    "step": "12",
    "param": 228,
}


@pytest.mark.parametrize(
    "func",
    [
        functions._mean,
        functions._std,
        functions._maximum,
        functions._minimum,
    ],
)
def test_multi_arg(func):
    arr = []
    for x in range(1, 5):
        request["number"] = x
        arr.append(retrieve("mars", request))
    func(*arr)


@pytest.mark.parametrize(
    "func",
    [
        functions._add,
        functions._subtract,
        functions._multiply,
        functions._divide,
    ],
)
def test_two_arg(func):
    arr = []
    for x in range(1, 3):
        request["number"] = x
        arr.append(retrieve("mars", request))
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
    functions.threshold(config, retrieve("mars", request))


def test_extreme():
    clim = retrieve(
        "mars",
        {
            "class": "od",
            "domain": "g",
            "expver": "0001",
            "levtype": "sfc",
            "param": 228,
            "date": "20230911",
            "time": "00",
            "stream": "efhs",
            "type": "cd",
            "step": "0-24",
            "quantile": [f"{x}:100" for x in range(101)],
            "interpolate": {
                "grid": "O640",
                "intgrid": "none",
                "legendre-loader": "shmem",
                "matrix-loader": "file-io",
            },
        },
    )
    base_request = request.copy()
    base_request["interpolate"] = {
        "grid": "O640",
        "intgrid": "none",
        "legendre-loader": "shmem",
        "matrix-loader": "file-io",
    }

    # Control
    base_request["type"] = "cf"
    functions.efi(clim, retrieve("mars", base_request), 0.0001, 2, control=True)

    # Ensemble
    ens = FieldList()
    for x in range(1, 5):
        base_request["type"] = "pf"
        base_request["number"] = x
        ens += retrieve("mars", base_request)

    functions.efi(clim, ens, 0.0001, 2)
    functions.sot(clim, ens, 90, 0.0001, 2)


def test_wind_speed():
    new_request = request.copy()
    new_request["param"] = [165, 166]
    functions.wind_speed(retrieve("mars", new_request))

    new_request["step"] = [0, 12, 24]
    with pytest.raises(Exception):
        functions.wind_speed(retrieve("mars", new_request))


def test_filter():
    arr = []
    for i in range(1, 3):
        request["number"] = i
        arr.append(retrieve("mars", request))
    functions.filter("<", 2, arr[0], arr[1], 0)
