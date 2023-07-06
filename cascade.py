import randomname
import numpy as np
import cupy as cp
import copy



class Task:

    def __init__(self, graph, definition):
        self.function = definition[0]
        self.dependencies = []
        self.input = []
        self.output = []
        self.dependents = []
        self.status = "NOT_STARTED"

        for i in range(1, len(definition)):
            self.dependencies.append(graph.create_or_get_task(definition[i]))

        for dependency in self.dependencies:
            dependency.dependents.append(self)

    def get_inputs(self):
        # receive inputs from each dependency task
    
    def execute(self):
        self.function(*self.input)

    def notify(self):
        # tell the manager that we have completed and output is ready

    def put_outputs(self):
        # send outputs to each dependent task

    def run(self):
        self.get_inputs()
        self.execute()
        self.notify()
        self.put_outputs()


class Graph:

    def __init__(self, from_dict):
        self.computations = copy.deepcopy(from_dict)
        self.tasks = {}

    def compute(self, key):

        self.create_or_get_task(key)

        root_tasks = []
        for t in self.tasks:
            if len(t.dependencies) == 0:
                root_tasks.append(t)



    def create_or_get_task(self, key):
        return self.computations.get(key, Task(self, self.computations[key]))


def read():
    # mimics a data creator (e.g. read from storage)
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    return a, b


def max(a):
    xp = cp.get_array_module(a)
    xp.max(a)
    return a


def main():

    # data objects, contains a name (generated) as well as information about the data size

    # build a graph which says which tasks produce which data objects and which consume it
    #  - tasks also contain execution information (GPU/CPU, execution times, etc.)

    # create a dictionary (? how to handle multiple outputs) of data objects

    # call get() with a set [] of output data objects to create

    G = Graph({
        "a": (read),
        "m": (max, "a")
        })

    G.compute("m")


main()
