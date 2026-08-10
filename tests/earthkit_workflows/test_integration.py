import earthkit.data
import numpy as np

from cascade.low.core import JobInstanceRich
from cascade.main import run_locally
from earthkit.workflows.compilers import graph2job
from earthkit.workflows.fluent import PayloadBuildingContext, create_task_instance, from_source


def earthkit_source(name: str, requests: list[dict], **kwargs) -> earthkit.data.SimpleFieldList:
    fieldlist = earthkit.data.SimpleFieldList()
    for request in requests:
        fieldlist += earthkit.data.from_source(name, request=request, **kwargs).to_fieldlist()  # type:ignore[unresolved-attribute]
    print(fieldlist.ls())
    return fieldlist


with PayloadBuildingContext(environment=["earthkit-data", "ecmwf-opendata", "polytope-client"]):
    action = from_source(
        np.array(
            [
                create_task_instance(
                    earthkit_source,
                    static_input_ps=[
                        [
                            "polytope",
                            [
                                {
                                    "class": "od",
                                    "stream": "oper",
                                    "param": "2t",
                                    "type": "fc",
                                    "levtype": "sfc",
                                    "date": "20260519",
                                    "time": "00",
                                    "step": [0, 6],
                                },
                                {
                                    "class": "od",
                                    "stream": "enfo",
                                    "param": "2t",
                                    "type": "pf",
                                    "levtype": "sfc",
                                    "date": "20260519",
                                    "time": "00",
                                    "step": [0, 6],
                                    "number": [1, 2],
                                },
                            ],
                        ],
                    ],
                )
            ]
        )
    )
    action = (
        action.expand(("number", [0, 1, 2]), ("number", [0, 1, 2]), backend_kwargs={"method": "isel"})
        .expand(("step", [0, 6]), ("step", [0, 1]), backend_kwargs={"method": "isel"})
        .mean(dim="number")
    )

if __name__ == "__main__":
    from earthkit.workflows.visualise import visualise

    visualise(action.graph(), "graph.html")
    run_locally(JobInstanceRich(jobInstance=graph2job(action.graph()), checkpointSpec=None), hosts=1, workers=1)
