#!/usr/bin/env python
# coding: utf-8

# ### Novel observations
# * Controller reports available as well, showing that a lot of time is spent in the `act` phase of sending (sequentially!) the commands to the hosts. I realize now that the `act` is blocking, so the transmit are effectively serialized at the controller! # noqa: E501
# * Redundant local transports mostly vanished -- there are times when the planner decides in a single step that a dataset is needed at two workers on a host so issues two transmit commands. We could thus replace the redundant sent by idle time, to save network etc. It happens only 3 times out of 55 in the 2,2 scenario # noqa: E501
#
# The `lA` logs here represent the code/measurements _before_ transmits were reworked to happen async, the `lB` the _after_.
# The `act` phase duration has shortened considerably, but the overall duration has increased -- possibly due to increased contention, due to introduction of locks, etc. But the overall amount of transmits has stayed roughly the same (even dripped a tiny bit). In particular, duration of the longest transmit has increased 4 times in the 2-host 2-worker scenario, **from 1 second to 4 seconds**. During that time, both sides of the transmit were doing other things as well (transmitting other datasets, computing tasks, etc). # noqa: E501
#
# ### Next steps
# * Rework the client to send asynchronously
# * Fuse the multi-transmit
# * When picking transmit, prefer local ones
# * Create a tooling for worker's timeline visualisation/exploration, to understand the contention
#   * Possibly parametrize the contention: how many concurrent transmits to allow, whether to allow transmits concurrent to task, pick least busy worker for transmits...
#

# In[1]:


import numpy as np
import pandas as pd

# In[2]:


def fixWorker(df):
    rows = df["host"] != "controller"
    df.loc[rows, "worker"] = df.loc[rows, "host"] + ":" + df.loc[rows, "worker"]


def readAll(base):
    c = pd.read_json(f"{base}.controller.jsonl", lines=True)
    t = pd.read_json(f"{base}.tasks.jsonl", lines=True)
    d = pd.read_json(f"{base}.datasets.jsonl", lines=True)
    if "M" in base:
        fixWorker(t)
        if d.shape[0] > 0:
            fixWorker(d)
    return c, t, d


# In[3]:


f1c, f1t, f1d = readAll("lA_F_1")
f4c, f4t, f4d = readAll("lA_F_4")
m14c, m14t, m14d = readAll("lA_M_1_4")
m41c, m41t, m41d = readAll("lA_M_4_1")
m22c, m22t, m22d = readAll("lA_M_2_2")
# after making the transmit non-blocking
n14c, n14t, n14d = readAll("lB_M_1_4")
n41c, n41t, n41d = readAll("lB_M_4_1")
n22c, n22t, n22d = readAll("lB_M_2_2")


# In[4]:


def fixMode(df):
    rows = ~df.dataset.isna()
    proj = df[rows & ~df["mode"].isna()].set_index(["dataset", "worker"])["mode"]
    lookup = proj[~proj.index.duplicated(keep="last")]
    return (
        df.set_index(["dataset", "worker"])
        .drop(columns="mode")
        .join(lookup)
        .reset_index()
    )


def fmn(n):  # TODO set some central
    return f"{n:.3e}"


def ensureColumns(df, columns):
    for column in columns:
        if column not in df.columns:
            df = df.assign(**{column: np.nan})
    return df


def analyzeController(df):
    print(f"phases: {df.shape[0]}")
    print(f"total waits duration: {fmn(df.waitDuration.sum())}")
    print(f"total act duration: {fmn(df.actDuration.sum())}")
    print(
        f"transmits issued: {df.actionsTransmit.sum()}, transmits received: {df.eventsTransmited.sum()}"
    )
    print(f"busy-during-wait: {fmn((df.busyWorkers * df.waitDuration).sum())}")


def transmitDurations(df):
    datasets = fixMode(df)
    durations = datasets.pivot(
        index=["dataset", "worker", "mode"], columns=["action"], values=["at"]
    )
    durations.columns = [name[1][len("transmit") :] for name in durations.columns]
    durations = durations.reset_index()
    localFix = durations["mode"] == "local"
    durations.loc[localFix, "Started"] = durations.loc[localFix, "Finished"]
    durations.loc[localFix, "Loaded"] = durations.loc[localFix, "Finished"]
    durations = durations.assign(total=durations.Finished - durations.Planned)
    durations = durations.assign(commDelay=durations.Started - durations.Planned)
    durations = durations.assign(loadDelay=durations.Loaded - durations.Started)
    durations = durations.assign(transmitDelay=durations.Finished - durations.Loaded)
    return durations


def taskDurations(df):
    tasks = df[~df.task.isna()]
    durations = tasks.pivot(index=["task", "worker"], columns=["action"], values=["at"])
    durations.columns = [name[1][len("task") :] for name in durations.columns]
    durations = durations.reset_index()
    durations = durations.assign(total=durations.Finished - durations.Planned)
    durations = durations.assign(commDelay=durations.Enqueued - durations.Planned)
    durations = durations.assign(queueDelay=durations.Started - durations.Enqueued)
    durations = durations.assign(loadDelay=durations.Loaded - durations.Started)
    durations = durations.assign(runtimes=durations.Finished - durations.Loaded)
    durations = durations.assign(onWorker=durations.Finished - durations.Enqueued)
    return durations


def analyzeTransmits(df):
    durations = transmitDurations(df)
    print(f"total transmit duration: {fmn(durations.total.sum())}")
    print(" *** ")
    print(f"mode counts: {durations['mode'].value_counts()}")
    print(
        f"per-mode transmit duration: {durations[['mode', 'total']].groupby('mode').sum()}"
    )
    print(" *** ")
    print(f"total comm delay: {fmn(durations.commDelay.sum())}")
    print(f"mean comm delay: {fmn(durations.commDelay.mean())}")
    print(f"max comm delay: {fmn(durations.commDelay.max())}")
    print(" *** ")
    remotes = durations.query("mode == 'remote'")
    print(f"total load delay: {fmn(remotes.loadDelay.sum())}")
    print(f"mean load delay: {fmn(remotes.loadDelay.mean())}")
    print(f"max load delay: {fmn(remotes.loadDelay.max())}")
    print(" *** ")
    print(f"total transmit delay: {fmn(remotes.transmitDelay.sum())}")
    print(f"mean transmit delay: {fmn(remotes.transmitDelay.mean())}")
    print(f"max transmit delay: {fmn(remotes.transmitDelay.max())}")
    print(" *** ")


def analyzeTasks(df):
    durations = taskDurations(df)
    print(f"total task duration: {fmn(durations.total.sum())}")
    print(" *** ")
    print(
        f"total task duration per worker: {durations.groupby('worker').onWorker.agg(['mean', 'sum'])}"
    )
    print(" *** ")
    print(f"total comm delay: {fmn(durations.commDelay.sum())}")
    print(f"mean comm delay: {fmn(durations.commDelay.mean())}")
    print(f"max comm delay: {fmn(durations.commDelay.max())}")
    print(" *** ")
    print(f"total queue delay: {fmn(durations.queueDelay.sum())}")
    print(f"mean queue delay: {fmn(durations.queueDelay.mean())}")
    print(f"max queue delay: {fmn(durations.queueDelay.max())}")
    print(" *** ")
    print(f"total runtime delay: {fmn(durations.runtimes.sum())}")


# In[5]:


analyzeController(f1c)


# In[6]:


analyzeController(f4c)


# In[7]:


analyzeController(m14c)


# In[8]:


analyzeController(n14c)


# In[9]:


analyzeController(m22c)


# In[10]:


analyzeController(n22c)


# In[11]:


analyzeController(m41c)


# In[12]:


analyzeController(n41c)


# In[13]:


analyzeTransmits(m22d)


# In[14]:


analyzeTransmits(n22d)


# In[15]:


analyzeTransmits(m41d)


# In[16]:


analyzeTransmits(n41d)


# In[20]:


Dn22d = transmitDurations(n22d)
Dm22d = transmitDurations(m22d)


# In[24]:


Dn22d.sort_values(by="transmitDelay", ascending=False)[:5]


# In[25]:


Dm22d.sort_values(by="transmitDelay", ascending=False)[:5]
