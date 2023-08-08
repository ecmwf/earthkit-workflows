import numpy as np
from ppgraph import Graph, Node
import randomname

def _mean(x):
    sum = 0
    count = 0
    for m in x:
        sum += m
        count += 1
    return sum/count

# -----------------------

class SubTask():
    def __init__(self, root, func):
        self.func = func
        self.next = None
        self.root = root

    def execute(self, input):
        return self.func(input)

    def then(self, func):
        self.next = SubTask(self.root, func)
        return self.next
    
    def foreach(self, func):
        self.next = SubTask(self.root, lambda x: [func(y) for y in x])
        return self.next
    
    def first(self):
        self.next = SubTask(self.root, lambda x: x[0] )
        return self.next

    def mean(self):
        self.next = SubTask(self.root, lambda x: _mean(x) )
        return self.next
    
    def where(self, func):
        self.next = SubTask(self.root, lambda x: [y for y in x if func(y)])
        return self.next

    # def broadcast(self, *args):
    #     self.next = args[0] # next should be a list
    #     return self.next

# -----------------------

    def graph(self, input=None):
        return Graph([self.root._graph(None)])

    def _graph(self, previous):
        # create a node in the graph
        if previous is not None:
            node = Node(randomname.generate(), ["foo"], payload=self.func, input=previous)
        else:
            node = Node(randomname.generate(), ["foo"], payload=self.func)

        if self.next is None:
            return node

        return self.next._graph(node.foo)

    def now(self, input=None):
        return self.root._now(input)

    def _now(self, input=None):
        result = self.execute(input)
        print(result)
        if self.next is None:
            return result
        return self.next._now(result)
    

class MultiToMultiTask(SubTask):
    def _graph(self, previous):
        # create a node in the graph
        if previous is not None:
            node = Node(randomname.generate(), ["foo"], payload=self.func, input=previous)
        else:
            node = Node(randomname.generate(), ["foo"], payload=self.func)

        if self.next is None:
            return node

        return self.next._graph(node.foo)

# -----------------------

class RootTask(SubTask):
    def __init__(self, func, count=1):
        super().__init__(self, func)

# -----------------------

def source_array():
    return RootTask(lambda x: [i for i in x])

def source_scalar():
    return RootTask(lambda x: x)

pyfdb=None

def read_fdb():
    return RootTask(lambda x: [f for f in pyfdb.FDB.read(x)])

def instant_prob(field):
    scalar = np.mean(field)
    return scalar

def threshold(scalar):
    return scalar > 0.5

######


task = source_array().foreach(lambda x : x + 1).where(lambda x: x > 2).mean().then(lambda x: x + 1)

graph = task.graph()

from ppgraph import pyvis
net = pyvis.to_pyvis(graph, notebook=True)
net.show("test.html")


# cascade = Cascade()

# windows = [[0, 20], [10, 30], [20, 40], [30, 50]]

# for window in windows:
#     start = window[0]
#     end = window[1]

#     task = read_fdb(f"request: {start} {end}").foreach(spectral_transform).groupby(lambda x: x.step).mean().threshold("> 2")
#     cascade.add(task)


# cascade.schedule()


# array = [ 0, 1, 2, 3 ]
# print(task.now(array))

# array2 = [ 1, 0, 1, 2, 3 ]
# print(task.now(array2))


# task = source_scalar().then(lambda x : x + 1).then(lambda x: x + 1)
# print(task.now(1))


# # example
# task = read_fdb().foreach(instant_prob).foreach(threshold, "< 2").mean().write_fdb()
# task.now("step=1/to/10,field=temperature", 50)
# task.now([1, 2, 3, 4])
# # should be able to build a task graph from this...