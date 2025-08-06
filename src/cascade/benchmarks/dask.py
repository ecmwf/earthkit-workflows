import dask.dataframe as dd
from dask._task_spec import convert_legacy_graph

from cascade.low.core import JobInstance
from cascade.low.dask import graph2job


def get_job(job: str) -> JobInstance:

    if job == "add":

        def add(x, y):
            result = x + y
            print(f"da {result=}")
            return result

        dl = {"a": 1, "b": 2, "c": (add, "a", "b")}
        dn = convert_legacy_graph(dl)
        job = graph2job(dn)
        job.ext_outputs = [
            dataset for task in job.tasks for dataset in job.outputs_of(task)
        ]
        return job
    elif job == "groupby":
        df = dd.DataFrame.from_dict({"x": [0, 0, 1, 1], "y": [1, 2, 3, 4]})
        df = df.groupby("x").sum()
        job = graph2job(df.__dask_graph__())
        job.ext_outputs = [
            dataset for task in job.tasks for dataset in job.outputs_of(task)
        ]
        return job
    else:
        raise NotImplementedError(job)
