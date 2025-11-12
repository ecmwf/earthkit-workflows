"""We will craft the same job which ultimately allows the same customizability,
derives much more + has more defaults, but still operates on individual task level
"""

from t00_execute import run_job
from t01_low import task1_callable, task2_callable, task3_callable

from cascade.low.builders import JobBuilder, TaskBuilder

# we dont need to give the input/output schemata anymore, it is automatically derived
task1 = TaskBuilder.from_callable(task1_callable, environment=["xarray"])
task2 = TaskBuilder.from_callable(task2_callable, environment=["xarray"])
task3 = TaskBuilder.from_callable(task3_callable, environment=["numpy"])

job = (
    (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2", task2)
        .with_edge("task1", "task2", "d")
        .with_node("task3", task3)
        .with_edge("task2", "task3", "a")
        .with_output("task3")
    )
    .build()
    .get_or_raise()
)  # consistency validation happening here

if __name__ == "__main__":
    print(run_job(job))
    # identical output as before, actually the JobInstance itself is the same too
