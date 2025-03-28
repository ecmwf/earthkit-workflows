#!/usr/bin/env python
# coding: utf-8

# ### Major mysteries
# * Why is the wall time inside the tasks so much higher for m1_4 and f4, but less so for m2_2/m4_1? What sort of cpu contention we have?
# * What to measure so that we have a clear indicator for m2_2/m4_1 taking much longer than m1_4/f4? How to separate m1_4 and f4 cleanly?
#
# ### Other data (extracted manually from logs)
#  - number of controller iterations (plan-act-await), number of events
#    - f4 -> 198; 223
#    - f1 -> 135; 135 (no transmits -> event count == iteration count == graph size)
#    - m_4_1 -> 66; 135
#    - m_2_2 -> 65; 165
#    - m_1_4 -> 103; 232
#    - there is much more event queuing in the multihost scenario, presumably because of the comm delay. Meaning the controller decides with less granularity
#    - the event counts are underreported here because of a *bug* (remote transfer doesnt succ mark the input as present)
#    - there is about the same number of data transmits in f4 and m1_4, meaning the final schedules are not that much different
#
# ### Next steps
#  - (feature) replace remote transfers with local ones when it is possible
#  - (bugfix) report remote transfer events correctly
#  - (feature) extend tracing for controller phase, event batch size, phase duration, total runtime

# In[1]:


import numpy as np
import pandas as pd

# In[19]:


def fixWorker(df):
    rows = df["host"] != "controller"
    df.loc[rows, "worker"] = df.loc[rows, "host"] + ":" + df.loc[rows, "worker"]


# In[152]:


f1 = pd.read_json("lA_F_1.jsonl", lines=True)
f4 = pd.read_json("lA_F_4.jsonl", lines=True)
m1_4 = pd.read_json("lA_M_1_4.jsonl", lines=True)
fixWorker(m1_4)
m4_1 = pd.read_json("lA_M_4_1.jsonl", lines=True)
fixWorker(m4_1)
m2_2 = pd.read_json("lA_M_2_2.jsonl", lines=True)
fixWorker(m2_2)


# In[283]:


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


def ensureColumns(df, columns):
    for column in columns:
        if column not in df.columns:
            df = df.assign(**{column: np.nan})
    return df


def transmitDurations(df):
    df = fixMode(df)
    datasets = df[~df.dataset.isna()].drop(columns="task")
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
    durations = durations.assign(runtimes=durations.Finished - durations.Started)
    durations = durations.assign(onWorker=durations.Finished - durations.Enqueued)
    return durations


def fmn(n):
    return f"{n:.3e}"


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


# In[261]:


analyzeTasks(m2_2)


# In[262]:


analyzeTasks(m1_4)


# In[263]:


analyzeTasks(m4_1)


# In[264]:


task_f1 = taskDurations(f1)
analyzeTasks(f1)


# In[265]:


task_f4 = taskDurations(f4)
analyzeTasks(f4)


# In[248]:


taskCompareF1F4 = (
    task_f1.set_index(["task"])[["total"]]
    .rename(columns={"total": "total1"})
    .join(task_f4.set_index(["task"])[["total"]].rename(columns={"total": "total4"}))
)
taskCompareF1F4 = taskCompareF1F4.assign(
    dif=taskCompareF1F4.total4 - taskCompareF1F4.total1
)
taskCompareF1F4 = taskCompareF1F4.assign(
    rel=taskCompareF1F4.dif / taskCompareF1F4.total4
)
taskCompareF1F4.sort_values(by="rel")


# In[251]:


taskCompareF1F4.sort_values(by="dif")[-10:]


# ## Task Takeaways:
# * There is a big difference between f1 and f4 in pure runtimes of tasks, 17e9 vs 28e9, suggesting some contention happening
#   * Comparing individual tasks, we see only small relative/abs differences in concats and disk-accessing retrieves, but big in compute intensive sot or efi, suggesting there is some CPU contention
#   * The difference is also visible for m scenarios -- m1_4 is expectedly like f4, but m2_2 and m4_1 are 20e9 being thus closer to f1. It could be that there is less overlap in those scenarios, as the scheduling is more gappy due to interleaved http comms? # noqa:E501
# * Queue delay exhibits no real difference over f/m scenarios
# * Comm delays are 1e7 for f scenarios, 1e8 for m4_1, and 1e9 for m2_2 and m1_4 -- m4_1 being midway looks more like a glitch
# * m2_2 is showing a slight disbalance of one worker being less utilised than the others, all others look balanced

# In[284]:


analyzeTransmits(f4)


# In[285]:


trans_m2_2 = transmitDurations(m2_2)
analyzeTransmits(m2_2)


# In[289]:


fmn(trans_m2_2.query("mode == 'redundant'").total.sum())


# In[287]:


analyzeTransmits(m4_1)


# In[288]:


trans_m1_4 = transmitDurations(m1_4)
analyzeTransmits(m1_4)


# ## Transmit Takeaways
# * The number of redundant transfers is low, just 8, in the 2-2 scenario. However, they still contributed 1e9 to the total runtime!
# * Much more remote than local transfers in the 2-2 scenario -- 166 vs 27
# * Mean comm delay for m1_4 is 9e6 whereas for m4_1 its 1e7 -- suggesting number of hosts is not that important on this front

# In[274]:


print(trans_m2_2[["dataset"]].value_counts().sum())
trans_m2_2[["dataset"]].value_counts()


# In[275]:


print(trans_m1_4[["dataset"]].value_counts().sum())
trans_m1_4[["dataset"]].value_counts()
