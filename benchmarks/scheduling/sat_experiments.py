from dataclasses import dataclass

from ortools.sat.python import cp_model

TaskId = str

# NOTE read more docs, eg,
# https://github.com/google/or-tools/blob/stable/ortools/sat/docs/boolean_logic.md
# https://github.com/google/or-tools/blob/stable/ortools/sat/docs/channeling.md
# and see if number of constraints etc can be brought down, the solver tuned, ...


@dataclass
class SimplifiedGraph:
    transport_overhead: dict[TaskId, int]
    input_edges: list[tuple[TaskId, TaskId]]
    durations: dict[TaskId, int]


def simpletree(depth: int, branching: int, duration: int = 2) -> SimplifiedGraph:
    transport_overhead = {}
    input_edges = []
    durations = {}

    stack = ["t"]
    transport_overhead["t"] = 1
    durations["t"] = 1
    while stack:
        task = stack.pop(-1)
        for j in range(branching):
            child = task + str(j)
            durations[child] = duration
            transport_overhead[child] = 1
            input_edges.append((child, task))
            if len(child) <= depth:
                stack.append(child)

    return SimplifiedGraph(transport_overhead, input_edges, durations)


@dataclass
class Problem:
    model: cp_model.CpModel
    taskIntervals: dict[TaskId, cp_model.IntervalVar]
    worker2task: dict[TaskId, str]
    end: cp_model.IntVar
    workers: list[str]
    tasks: list[str]


def graph2sat(g: SimplifiedGraph, workers: list[str]):
    model = cp_model.CpModel()

    max_total = sum(g.durations.values())
    tasks = list(g.durations.keys())
    # task@worker -- each task at exactly one
    w2t = {}
    for task in tasks:
        for worker in workers:
            w2t[(task, worker)] = model.new_int_var(0, 1, f"{task}@{worker}")
        model.add(sum(w2t[(task, worker)] for worker in workers) == 1)
    # start and end of task -- respect durations
    tInt = {}
    for task, duration in g.durations.items():
        start = model.new_int_var(0, max_total, f"{task}_start")
        end = model.new_int_var(0, max_total, f"{task}_end")
        interv = model.new_interval_var(start, duration, end, f"{task}_interval")
        tInt[task] = (start, end, interv)
    # helper variables -- coscheduling, overlap
    isCoscheduled = {}
    isRelated = {}
    for i in range(len(tasks) - 1):
        for j in range(i + 1, len(tasks)):
            cosch = model.new_int_var(0, 1, f"{tasks[i]}~{tasks[j]}")
            isCoscheduled[(tasks[i], tasks[j])] = cosch
            isCoscheduled[(tasks[j], tasks[i])] = cosch
            loc = []
            for worker in workers:
                coschLoc = model.new_int_var(0, 1, f"{tasks[i]}~{tasks[j]}@{worker}")
                loc.append(coschLoc)
                model.add_multiplication_equality(
                    coschLoc, [w2t[(tasks[i], worker)], w2t[(tasks[j], worker)]]
                )
            # coscheduled variable constraint
            model.add(sum(loc) == cosch)
            # respect worker parallelism

            # NOTE constraining no overlap is not supported!
            # model.add_no_overlap([tInt[tasks[i]][2], tInt[tasks[j]][2]]).only_enforce_if(cosch)

            # dont start two tasks at the very same time if coscheduled
            # model.add(tInt[tasks[i]][0] != tInt[tasks[j]][0]).only_enforce_if(cosch) # redundant due to latter?

            # declare isAfter/isBefore helpers, if coscheduled enforce one
            isAfter = model.new_bool_var(f"{tasks[i]}>={tasks[j]}")
            model.add(tInt[tasks[i]][0] >= tInt[tasks[j]][1]).only_enforce_if(isAfter)
            isBefore = model.new_bool_var(f"{tasks[i]}<={tasks[j]}")
            model.add(tInt[tasks[j]][0] >= tInt[tasks[i]][1]).only_enforce_if(isBefore)
            isRelated[(tasks[i], tasks[j])] = (isAfter, isBefore)
            model.add_bool_or([isAfter, isBefore]).only_enforce_if(cosch)

    # respect input edges
    for head, tail in g.input_edges:
        t = g.transport_overhead[head]
        maybe_overhead = t - t * isCoscheduled[(head, tail)]
        model.add(tInt[head][1] + maybe_overhead <= tInt[tail][0])

    end = model.new_int_var(0, max_total, "end")
    for task in tasks:
        model.add(end >= tInt[task][1])
    model.minimize(end)
    print(model.validate())

    return Problem(
        model=model,
        taskIntervals=tInt,
        worker2task=w2t,
        end=end,
        workers=workers,
        tasks=tasks,
    )


def solve(problem: Problem):
    solver = cp_model.CpSolver()
    status = solver.solve(problem.model)

    print(f"status: {solver.status_name(status)}")
    for task in problem.tasks:
        for worker in problem.workers:
            if solver.value(problem.worker2task[(task, worker)]) == 1:
                target = worker
        print(
            f"{task=} => from {solver.value(problem.taskIntervals[task][0])} to {solver.value(problem.taskIntervals[task][1])} at {target}"
        )
    print(f"end = {solver.value(problem.end)}")


if __name__ == "__main__":
    g2 = simpletree(2, 2)
    problem = graph2sat(g2, ["w1", "w2"])
    solve(problem)
