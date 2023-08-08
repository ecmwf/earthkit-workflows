import randomname
from ppgraph import Node, Graph
from typing import List

def _mean(x):
    count = 0
    sum = 0
    for m in x:
        sum += m
        count += 1
    return sum/count
        

class Action:
    def __init__(self, func, root=None, count=1):
        self.func = func
        self.next = None
        self.root = root or self
        self.count = count

    # -- Generators

    def mean(self):
        self.next = MultiToSingleAction(_mean, self.root)
        return self.next
    
    def then(self, func):
        self.next = SingleToSingleAction(func, self.root)
        return self.next
    
    def foreach(self, func):
        self.next = MultiToMultiAction(func, self.root)
        return self.next
    
    def read(self, size):
        self.next = SingleToMultiAction(lambda x: [x for i in range(size)], self.root, size)
        return self.next
    
    # -- Graph Generation
    
    def graph(self) -> List[Node]:
        return self.root._graph(None)

    def _graph(self, previous: List[Node]) -> List[Node]:
        # root node does nothing, just returns the next node
        return self.next._graph(None)
        

class SingleToSingleAction(Action):
    
    def _graph(self, previous):

        node = Node(randomname.generate(), payload=self.func)

        if previous is not None:
            assert len(previous) == 1
            node.inputs[randomname.generate()] = previous[0].get_output()

        if self.next is None:
            return [node]
        return self.next._graph([node])

class MultiToMultiAction(Action):
    
    def _graph(self, previous):

        assert previous is not None

        nodes = []
        for p in previous:
            node = Node(randomname.generate(), payload=self.func, input=p)
            nodes.append(node)
        
        if self.next is None:
            return nodes
        return self.next._graph(nodes)
        
    
class SingleToMultiAction(Action):

    def _graph(self, previous):

        # create nodes in the graph
        nodes = []
        for c in range(self.count):
            nodes.append( Node(randomname.generate(), payload=self.func) )
            if previous is not None:
                assert len(previous) == 1
                nodes[c].inputs[randomname.generate()] = previous[0].get_output()
        
        if self.next is None:
            return nodes

        return self.next._graph(nodes)        
    
class MultiToSingleAction(Action):
    def _graph(self, previous):

        node = Node(randomname.generate(), payload=self.func)

        if previous is not None:
            for i,p in enumerate(previous):
                node.inputs[randomname.generate()] = p.get_output()

        if self.next is None:
            return [node]
        
        return self.next._graph([node])


root = Action(None)

windows = [[0, 20], [10, 30], [20, 40], [30, 50], [40, 60], [50, 70], [60, 80]]


g = Graph([])

for window in windows:
    root = Action(None)
    window_graph = root.read(size=20).foreach(lambda x: x+1).mean().graph()

# Cascade::read().foreach().mean().then()
# Cascade::merge().mean().then().graph()

# orderby
# groupby
# take
# where
# sum/max/min/mean
# join (into tuples)

# Cascade::read().joinread()





from ppgraph import pyvis
pyvis.to_pyvis(g, notebook=True).show("test.html")