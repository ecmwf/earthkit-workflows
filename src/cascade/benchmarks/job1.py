# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""One particular job suite for benchmarking: prob, efi ensms

When executed as a module, downloads the datasets as local files

Controlled by env var params: JOB1_{DATA_ROOT, GRID, ...}, see below
"""

import os

import earthkit.data
from ppcascade.fluent import from_source
from ppcascade.utils.window import Range

from earthkit.workflows.fluent import Payload

# *** PARAMS ***

# eg "/ec/res4/hpcperm/ecm6012/gribs/casc_g01/"
data_root = os.environ["JOB1_DATA_ROOT"]
# 60, 120, 180, 240
END_STEP = int(os.environ["JOB1_END_STEP"])
# 10 to 50
NUM_ENSEMBLES = int(os.environ["JOB1_NUM_ENSEMBLES"])
# O320, O640 or O1280
GRID = os.environ["JOB1_GRID"]
DATE = "20241111"
CLIM_DATE = "20241110"

# ** JOB DEFINITIONS ***

files = [
    f"{data_root}/data_{number}_{step}.grib"
    for number in range(1, NUM_ENSEMBLES + 1)
    for step in range(0, END_STEP + 1, 3)
]
payloads = [
    Payload(
        lambda f: earthkit.data.from_source("file", f),
        (f,),
    )
    for f in files
]
inputs = from_source(
    [
        {
            "source": "fileset",
            "location": data_root + "/data_{number}_{step}.grib",
            "number": list(range(1, NUM_ENSEMBLES + 1)),
            "step": list(range(0, END_STEP + 1, 3)),
        }
    ]
)
climatology = from_source(
    [
        {
            "source": "fileset",
            "location": data_root + "/data_clim_{stepRange}.grib",
            "stepRange": ["0-24", "24-48"],  # list(range(0, END_STEP - 23, 24)),
        }
    ]
)


def get_prob():
    prob_windows = [Range(f"{x}-{x}", [x]) for x in range(0, END_STEP + 1, 24)] + [
        Range(f"{x}-{x + 120}", list(range(x + 6, x + 121, 6)))
        for x in range(0, END_STEP - 119, 120)
    ]
    return (
        inputs.window_operation("min", prob_windows, dim="step", batch_size=2)
        .ensemble_operation(
            "threshold_prob",
            comparison="<=",
            local_scale_factor=2,
            value=273.15,
        )
        .graph()
    )


def get_ensms():
    # Graph for computing ensemble mean and standard deviation for each time step
    return inputs.ensemble_operation("ensms", dim="number", batch_size=2).graph()


def get_efi():
    efi_windows = [
        Range(f"{x}-{x+24}", list(range(x + 6, x + 25, 6)))
        for x in range(0, END_STEP - 23, 24)
    ]
    return (
        inputs.window_operation("mean", efi_windows, dim="step", batch_size=2)
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
                "timeRangeIndicator": 3,
            },
        )
        .graph()
    )


# *** DATA DOWNLOADERS ***


def download_inputs():
    for number in range(1, NUM_ENSEMBLES + 1):
        for step in range(0, END_STEP + 1, 3):
            ekp = {
                "class": "od",
                "expver": "0001",
                "stream": "enfo",
                "date": DATE,
                "time": "00",
                "param": 167,
                "levtype": "sfc",
                "type": "pf",
                "number": number,
                "step": step,
                "grid": GRID,
            }
            data = earthkit.data.from_source("mars", **ekp)
            with open(f"{data_root}/data_{number}_{step}.grib", "wb") as f:
                data.write(f)


def download_climatology():
    for step in range(0, END_STEP - 23, 24):
        ekp = {
            "class": "od",
            "expver": "0001",
            "stream": "efhs",
            "date": CLIM_DATE,
            "time": "00",
            "param": 228004,
            "levtype": "sfc",
            "type": "cd",
            "quantile": [f"{x}:100" for x in range(101)],
            "step": f"{step}-{step+24}",
            "grid": GRID,
        }
        data = earthkit.data.from_source("mars", **ekp)
        with open(f"{data_root}/data_clim_{step}.grib", "wb") as f:
            data.write(f)


if __name__ == "__main__":
    download_inputs()
    download_climatology()
