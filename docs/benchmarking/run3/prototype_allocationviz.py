# In[1]:


import pyvis

# In[31]:


colors = {
    "h1": "#ff0000",
    "h2": "#00ff00",
    "h3": "#0000ff",
}


def add_node(g, l: str):
    task, worker = l.strip().rsplit(";", 1)
    host, _ = worker.split(":", 1)
    g.add_node(task, color=colors[host])


def add_inputs(g, l: str):
    task, *inputs = l.strip().split(";")
    for i in inputs:
        if i:
            g.add_edge(task, i)


# cat l1ctrl.txt  | grep 'action=taskPlanned' | sed 's/.*task=\([^;]*\).*worker=\([^;]*\).*/\1;\2/' > g1_task_host.csv
# cat l1*.txt  | grep 'about to start subgraph' | sed 's/.*name=.\(.*\).,.*wirings=\[\([^]]*\)\].*/\1;\2/' | sed "s/'Any')/@/g" | sed "s/@, /@;/g" | sed "s/Variable[^@]*task='\([^']*\)'[^@]*@/\1/g" > g1_task_inputs.csv # noqa: E501


def vizLogs(pref: str):
    g = pyvis.network.Network(notebook=True)

    t2h = open(f"{pref}_task_host.csv").readlines()
    t2i = open(f"{pref}_task_inputs.csv").readlines()
    for l in t2h:
        add_node(g, l)
    for l in t2i:
        add_inputs(g, l)

    return g.show("o.html", notebook=True)
