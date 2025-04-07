# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Intended to be run inside a jupyter notebook

Run `dirTaskLane(<path-to-your-logs-directory>)`
"""

import os

from bokeh.io import curdoc, output_notebook, show
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot, VSpan

from cascade.benchmarks.reporting import logParse

output_notebook()


def plotTaskLane(
    data: dict, scale, width, isBarController=False, allowFailed=False, vizTrans=True
):
    Td = data["task_durations"].dropna()
    nc = [
        "planned",
        "enqueued",
        "started",
        "loaded",
        "computed",
        "published",
        "completed",
    ]
    df = (Td[nc] / scale).astype(int)
    zero = df.planned.min()
    df = df - zero
    df = df.assign(worker=Td.worker)

    workerToLane = {
        e: i for i, e in enumerate(Td.worker.drop_duplicates().sort_values())
    }
    print(workerToLane)  # TODO use the labels in the if instead

    plot = Plot(
        title=None,
        width=width,
        height=500,
        min_border=0,
        toolbar_location=None,
    )  # TODO derive width from data

    def boxTask(left, right, color):
        source = ColumnDataSource(
            dict(y=df.worker.map(workerToLane), left=left, right=right)
        )
        glyph = HBar(y="y", right="right", left="left", height=0.5, fill_color=color)
        plot.add_glyph(source, glyph)

    def boxTransmit(worker, left, right, color):
        source = ColumnDataSource(
            dict(y=worker.map(workerToLane) - 0.5, left=left, right=right)
        )
        glyph = HBar(y="y", right="right", left="left", height=0.25, fill_color=color)
        plot.add_glyph(source, glyph)

    def barController(df, action, color):
        df = df.query(f"action == '{action}'")
        df = (df[["at"]] / scale).astype(int) - zero
        df = df.assign(width=2)
        source = ColumnDataSource(dict(x=df["at"], width=df.width))
        glyph = VSpan(x="x", line_width="width", line_color=color)
        plot.add_glyph(source, glyph)

    Ctrl = data["controller"]
    if isBarController:
        barController(Ctrl, "plan", "#00aa00")
        barController(Ctrl, "act", "#0000aa")
        barController(Ctrl, "wait", "#aa0000")
        barController(Ctrl, "shutdown", "#000000")

    boxTask(df.planned, df.enqueued, "#111111")
    boxTask(df.enqueued, df.started, "#dd1111")
    boxTask(df.started, df.loaded, "#1111dd")
    boxTask(df.loaded, df.computed, "#11dd11")
    boxTask(df.computed, df.published, "#1111dd")
    boxTask(df.published, df.completed, "#444444")

    if data["transmit_durations"].shape[0] > 0 and vizTrans:
        Rd = data["transmit_durations"].dropna()
        nc = ["planned", "started", "loaded", "received", "unloaded", "completed"]
        df = (Rd[nc] / scale).astype(int)
        df = df - zero
        df = df.assign(target=Rd.target + ".w0", source=Rd.source)
        boxTransmit(df.target, df.planned, df.received, "#ff1111")  # target waiting RED
        boxTransmit(
            df.target, df.received, df.unloaded, "#1111ff"
        )  # target memcpy BLUE
        boxTransmit(
            df.target, df.unloaded, df.completed, "#444444"
        )  # target callback GREY
        boxTransmit(
            df.source, df.planned, df.started, "#aaaa11"
        )  # ctrl2source comm delay YELLOW
        boxTransmit(df.source, df.started, df.loaded, "#1111aa")  # source memcpy BLUE
        boxTransmit(
            df.source, df.loaded, df.received, "#444444"
        )  # source memcpy + network GREY

    xaxis = LinearAxis()
    plot.add_layout(xaxis, "below")

    yaxis = LinearAxis()
    plot.add_layout(yaxis, "left")

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))  # type: ignore
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))  # type: ignore

    curdoc().add_root(plot)

    return show(plot)


def dirLogz(dname):
    files = [f"{dname}/{f}" for f in os.listdir(dname)]
    return logParse(files)


def dirTaskLane(dname, **kwargs):
    logz = dirLogz(dname)
    plotTaskLane(logz, 1e6, 4000, **kwargs)
