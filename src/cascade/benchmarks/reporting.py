# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Utilities for log parsing and performance reporting"""

import math
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from cascade.low.tracing import Microtrace, dataBegin, marker_logid, tracer_logid


@dataclass
class MicrotraceAccumulator:
    duration_ns_cnt: int = 0
    duration_ns_sum: int = 0

    def add(self, duration_ns: int) -> None:
        self.duration_ns_cnt += 1
        self.duration_ns_sum += duration_ns


def taskDurations(tasks: pd.DataFrame) -> pd.DataFrame:
    durations = tasks.pivot(index=["task", "worker"], columns=["action"], values=["at"])
    durations = durations.reset_index()
    durations.columns = ["task", "worker"] + [e[1] for e in durations.columns[2:]]  # type: ignore

    durations = durations.assign(total_e2e=durations.completed - durations.planned)
    durations = durations.assign(total_worker=durations.published - durations.enqueued)
    durations = durations.assign(comm2worker=durations.enqueued - durations.planned)
    durations = durations.assign(task_procstart=durations.started - durations.enqueued)
    durations = durations.assign(task_load=durations.loaded - durations.started)
    durations = durations.assign(task_callablerun=durations.computed - durations.loaded)
    durations = durations.assign(task_publish=durations.published - durations.computed)
    durations = durations.assign(comm2ctrl=durations.completed - durations.published)
    return durations


def transmitDurations(transmits: pd.DataFrame) -> pd.DataFrame:
    if transmits.shape[0] == 0:
        return transmits

    mode_fix = transmits[~transmits["mode"].isna()].set_index(["dataset", "target"])[
        "mode"
    ]
    lookup = mode_fix[~mode_fix.index.duplicated(keep="last")]
    transmits = (
        transmits.set_index(["dataset", "target"])
        .drop(columns="mode")
        .join(lookup)
        .reset_index()
    )

    # we'd have received and unloaded missing source
    source_fix = transmits[~transmits["source"].isna()].set_index(
        ["dataset", "target"]
    )["source"]
    lookup = source_fix[~source_fix.index.duplicated(keep="last")]
    transmits = (
        transmits.set_index(["dataset", "target"])
        .drop(columns="source")
        .join(lookup)
        .reset_index()
    )

    durations = transmits.pivot(
        index=["dataset", "target", "source", "mode"], columns=["action"], values=["at"]
    )
    durations.columns = [name[1] for name in durations.columns]  # type: ignore
    durations = durations.reset_index()

    durations = durations.assign(total=durations.completed - durations.planned)
    durations = durations.assign(comm2source=durations.started - durations.planned)
    durations = durations.assign(loadDelay=durations.loaded - durations.started)
    durations = durations.assign(comm2target=durations.received - durations.loaded)
    durations = durations.assign(unloadDelay=durations.unloaded - durations.received)
    durations = durations.assign(comm2ctrl=durations.completed - durations.unloaded)
    return durations


def logParse(files: Iterable[str]) -> dict[str, pd.DataFrame]:
    microtraces = {e: MicrotraceAccumulator() for e in Microtrace}
    marks: dict[str, list] = {
        "ctrl": [],
        "transmit": [],
        "task": [],
    }

    for fname in files:
        with open(fname) as f:
            for line in f.readlines():
                if tracer_logid in line:
                    data = line.split(dataBegin, 1)[1].strip()
                    kind, value = data.split("=", 1)
                    microtraces[Microtrace(kind)].add(int(value))
                if marker_logid in line:
                    data = line.split(dataBegin, 1)[1].strip()
                    mark = {}
                    for kv in data.split(";"):
                        k, v = kv.split("=", 1)
                        try:
                            v = int(v)  # type: ignore
                            mark[k] = v
                        except ValueError:
                            mark[k] = v
                    key, value = mark["action"].split("_", 1)
                    mark["action"] = value
                    marks[key].append(mark)

    microtraces_struct = [
        {
            "kind": k.value,
            "duration_ns_sum": v.duration_ns_sum,
            "duration_ns_cnt": v.duration_ns_cnt,
            "duration_ns_avg": (
                v.duration_ns_sum / v.duration_ns_cnt
                if v.duration_ns_cnt > 0
                else math.nan
            ),
        }
        for k, v in microtraces.items()
    ]

    rv = {
        "microtraces": pd.DataFrame(microtraces_struct),
        "controller": pd.DataFrame(marks["ctrl"]),
        "tasks": pd.DataFrame(marks["task"]),
        "transmits": pd.DataFrame(marks["transmit"]),
    }
    rv["task_durations"] = taskDurations(rv["tasks"])
    rv["transmit_durations"] = transmitDurations(rv["transmits"])
    return rv


if __name__ == "__main__":
    import sys

    print(logParse([sys.argv[1]]))
