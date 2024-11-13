"""
One particular job suite for benchmarking: prob, efi ensms
"""

from ppcascade.fluent import from_source
from cascade.fluent import from_source as c_from_source
from ppcascade.utils.window import Range
from ppcascade.utils.request import Request
from cascade.graph import deduplicate_nodes, Graph
from cascade.cascade import Cascade
from cascade.fluent import Payload
import earthkit.data

data_root = "~/warehouse/ecmwf/cascEx1"
END_STEP=60     # Can increase to a number divisible by 6 up to 240
NUM_ENSEMBLES=2 # Can increase up to 50

params = {
    "GRID": "O320",     # Valid: O320, O640 or O1280; mem goes up
    "DATE": "20241015",
    "CLIM_DATE": "20241014",
}

files = [
    f"{data_root}/data_{number}_{step}.grib"
    for number in range(1, NUM_ENSEMBLES + 1)
    for step in range(0, END_STEP + 1, 3)
]
payloads = [
    Payload(
        lambda f : earthkit.data.from_source("file", f),
        (f,),
    ) for f in files
]
inputs = from_source([
    {
        "source": "fileset", "location": data_root + "/data_{number}_{step}.grib",
        "number": list(range(1, NUM_ENSEMBLES + 1)),
        "step": list(range(0, END_STEP + 1, 3)),
    }
])
climatology = from_source([
    {
        "source": "fileset", "location": data_root + "/data_clim_{stepRange}.grib",
        "stepRange": ['0-24', '24-48'], # list(range(0, END_STEP - 23, 24)),
    }
])


def get_prob():
    prob_windows = [
        Range(f"{x}-{x}", [x]) for x in range(0, END_STEP + 1, 24)
    ] + [
        Range(f"{x}-{x + 120}", list(range(x + 6, x + 121, 6))) for x in range(0, END_STEP  - 119, 120)
    ]
    return (
        inputs
        .window_operation(
            "min",
            prob_windows,
            dim="step", batch_size=2)
        .ensemble_operation(
            "threshold_prob",
            comparison="<=",
            local_scale_factor=2,
            value= 273.15,
        )
        .graph()
    )

def get_ensms():
    # Graph for computing ensemble mean and standard deviation for each time step
    return (
        inputs
        .ensemble_operation("ensms", dim="number", batch_size=2)
        .graph()
    )

def get_efi():
    efi_windows = [
        Range(f"{x}-{x+24}", list(range(x+6, x+25, 6))) for x in range(0, END_STEP - 23, 24)
    ]
    return (
        inputs
        .window_operation(
            "mean",
            efi_windows,
            dim="step", batch_size=2)
        .ensemble_extreme(
            "extreme",
            climatology,
            efi_windows,
            sot=[10, 90],
            eps=1e-4,
            metadata={
                "edition": 1,
                "gribTablesVersionNo": 132,
                "indicatorOfParameter": 167,
                "localDefinitionNumber": 19,
                "timeRangeIndicator": 3
            }
        )
        .graph()
    )

def download_inputs():
    for number in range(1, NUM_ENSEMBLES + 1):
        for step in range(0, END_STEP + 1, 3):
            ekp = {
                "class": "od",
                "expver": "0001",
                "stream": "enfo",
                "date": params["DATE"],
                "time": "00",
                "param": 167,
                "levtype": "sfc",
                "type": "pf",
                "number": number,
                "step": step,
                "grid": params["GRID"],
            }
            data = earthkit.data.from_source("mars", **ekp)
            with open(f"{data_root}/data_{number}_{step}.grib", 'wb') as f:
                data.write(f)

def download_climatology():
    for step in range(0, END_STEP - 23, 24):
        ekp = {
            "class": "od",
            "expver": "0001",
            "stream": "efhs",
            "date": params["CLIM_DATE"],
            "time": "00",
            "param": 228004,
            "levtype": "sfc",
            "type": "cd",
            "quantile": [f"{x}:100" for x in range(101)],
            "step": f"{step}-{step+24}",
            "grid": params["GRID"],
        }
        data = earthkit.data.from_source("mars", **ekp)
        with open(f"{data_root}/data_clim_{step}.grib", 'wb') as f:
            data.write(f)

if __name__ == "__main__":
    download_inputs()
    download_climatology()
