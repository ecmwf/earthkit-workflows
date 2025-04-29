#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bokeh.io import curdoc, output_notebook, show
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot, VSpan

from cascade.benchmarks.reporting import logParse

output_notebook()


# In[34]:


def plotTaskLane(data: dict, scale, width):
    Td = data["task_durations"]
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

    Rd = data["transmit_durations"]
    nc = ["planned", "started", "loaded", "received", "unloaded", "completed"]
    df = (Rd[nc] / scale).astype(int)
    df = df - zero
    df = df.assign(target=Rd.target, source=Rd.source)
    boxTransmit(df.target, df.planned, df.received, "#111111")  # target waiting
    boxTransmit(df.target, df.received, df.unloaded, "#1111dd")  # target memcpy
    boxTransmit(df.target, df.unloaded, df.completed, "#444444")  # target callback
    boxTransmit(df.source, df.planned, df.started, "#111111")  # ctrl2source comm delay
    boxTransmit(df.source, df.started, df.loaded, "#1111dd")  # source memcpy
    boxTransmit(df.source, df.loaded, df.received, "#444444")  # source memcpy + network

    xaxis = LinearAxis()
    plot.add_layout(xaxis, "below")

    yaxis = LinearAxis()
    plot.add_layout(yaxis, "left")

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    curdoc().add_root(plot)

    return show(plot)


# In[87]:


l4prob = logParse(["l4prob.txt"])
plotTaskLane(l4prob, 1e6, 2000)


# In[41]:


l2_2prob = logParse(["l2_2.prob.txt"])
plotTaskLane(l2_2prob, 1e6, 2000)


# In[88]:


l1_4all = logParse(["l1_4.all.txt"])
plotTaskLane(l1_4all, 1e6, 6000)


# In[40]:


l2_2all = logParse(["l2_2.all.txt"])
plotTaskLane(l2_2all, 1e6, 6000)


# In[38]:


h1all = logParse(["h1.all.txt"])
plotTaskLane(h1all, 1e6, 6000)


# In[39]:


h2all = logParse(["h2.all.txt"])
plotTaskLane(h2all, 1e6, 10000)
