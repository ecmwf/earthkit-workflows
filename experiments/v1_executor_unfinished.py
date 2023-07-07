import randomname
import numpy as np
import cupy as cp
import ucp

################


class DataDef:
    def __init__(self, dtype=None, size=None, name=None):
        self.name = randomname.get_name()


class TaskDef:
    def __init__(self, function, inputs, outputs, name=None):
        self.name = function.__name__
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.outputs = outputs if isinstance(outputs, list) else [outputs]


################


class Task:
    def __init__(self, taskdef):
        self.taskdef = taskdef
        self.host = ucp.get_address(ifname="eth0")
        self.downstream = []  # data references
        self.upstream = []  # data references

    def execute(self):
        # call recv on all upstream data references
        # run the function
        # notify data available and call send on all downstream data references (dependencies can start)
        # notify and shutdown once complete (resources now free)
        pass


class Data:
    def __init__(self, datadef):
        self.datadef = datadef
        self.downstream = []  # task references
        self.upstream = None  # task reference

    def notify_manager(self):
        pass

    def send(self):
        # create a listener
        # send
        pass

    def recv(self):
        # create an endpoint
        # receive
        pass


################


class Graph:
    def __init__(self):
        self.taskdefs = {}
        self.tasks = {}

    def add(self, task):
        for o in task.outputs:
            self.taskdefs[o] = task

    def compute(self, datadef):
        assert datadef in self.taskdefs

        # Create tasks by picking from the library of taskdefs
        # Start from the final task and work upstream

        task = self.taskdefs[datadef]

        # when data is notified as available, check dependents of that data


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

    a = DataDef()
    m = DataDef()

    t1 = TaskDef(read, None, a)
    t2 = TaskDef(max, a, m)

    G = Graph()
    G.add(t1)
    G.add(t2)

    G.compute(m)


main()
