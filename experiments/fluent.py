import randomname
from ppgraph import Node, Graph
from typing import List
from ppgraph import pyvis
from itertools import product

def _mean(x):
    count = 0
    sum = 0
    for m in x:
        sum += m
        count += 1
    return sum/count

def _sum(x):
    sum = 0
    for m in x:
        sum += m
    return sum

def _read(x):
    # actually read the data one by one
    pass

def spectral_transform(x):
    return x

def _expand_request(req):
    metadata = {}
    for r in req.split(","):
        lhs, rhs = r.split("=")
        if "to" in rhs:
            rhs += "/by/1" # add default values
            frm,_,to,_,by,*_ = rhs.split("/")
            metadata[lhs] = [i for i in range(int(frm), int(to), int(by))]
        else:
            metadata[lhs] = [rhs]

    prod = list(product(*metadata.values()))
    product_with_keys = [{k: v for k, v in zip(metadata.keys(), values)} for values in prod]
    return product_with_keys
    
def _combine_dicts(dict_list):
    combined = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            if isinstance(value, list):
                combined[key].extend(value)
            else:
                combined[key].append(value)
    return dict(combined)


class Action:
    def __init__(self, func, root=None, count=1):
        self.func = func
        self.next = None
        self.root = root or self
        self.count = count

    # -- Generators

    def mean(self, key=""):
        self.next = MultiToSingleAction(_mean, self.root )
        return self.next
    
    def sum(self):
        self.next = MultiToSingleAction(_mean, self.root )
        return self.next
    
    def then(self, func):
        self.next = SingleToSingleAction(func, self.root)
        return self.next
    
    def foreach(self, func):
        self.next = MultiToMultiAction(func, self.root)
        return self.next
    
    def groupby(self, key):
        self.next = GroupingAction(key, self.root)
        return self.next
    
    def constant(self, value):
        self.next = SingleToSingleAction(lambda x: value, self.root)
        return self.next
    
    def repeat(self, count):
        self.next = MultiToMultiAction(lambda x: x, self.root, self.count * count)
        return self.next
    
    def write(self):
        self.next = SingleToSingleAction(lambda x: print(x), self.root)
        return self.next
    
    def threshold(self, threshold):
        self.next = MultiToMultiAction(lambda x: 1 if x > threshold else 0, self.root)
        return self.next
    
    # -- Graph Generation
    
    def graph(self) -> List[Node]:
        return Graph(self.root._graph(None))

    def _graph(self, previous: List[Node]) -> List[Node]:
        # root node does nothing, just returns the next node
        return self.next._graph(None)

class Cascade(Action):
    def __init__(self):
        super().__init__(None)

    def read(self, request):
        fields = _expand_request(request)
        self.next = SingleToMultiAction(lambda x: _read(x), self.root, len(fields))
        return self.next


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

class GroupingAction(Action):

    def __init__(self, key, root):
        super().__init__(None, root)
        if key == "step":
            self.count = 50 # hacky until we track metadata we can actually group on

    def _graph(self, previous):
        
        assert previous is not None

        nodes = []
        n = 0
        for p in previous:
            if n % self.count == 0:
                node = Node(randomname.generate(), payload=self.func)
                nodes.append(node)
            else:
                node = nodes[-1]
            node.inputs[str(n % self.count)] = p.get_output()
            n += 1
        
        if self.next is None:
            return nodes
        return self.next._graph(nodes)

# -------------------------------------------------------------------------------------

Cascade().read("stream=efhs,levtype=pl,level=850,step=0-240/by/12").foreach(spectral_transform).mean().write()

# Cascade().constant(1).repeat(50).sum().write()



# -------------------------------------------------------------------------------------

# windows = [[0, 20], [10, 30], [20, 40], [30, 50], [40, 60], [50, 70], [60, 80]]
window_ranges = [[120, 240], [168, 240]]
window_cascades = []
total_graph = Graph([])

for window in window_ranges:

    start = window[0]
    end = window[1]

    climatology = Cascade().read(f"stream=efhs,levtype=pl,level=850,step={start}-{end}")\
                    .foreach(spectral_transform)
    
    t850 = Cascade().read(f"date=...,levtype=pl,level=850,number=1/to/50,step={start}/to/{end}/by/12")\
                    .foreach(spectral_transform)\
                    .groupby("step")\
                    .foreach(lambda x: x - 2)\
                    .mean("step")\
                    .threshold("< -2")\
                    .mean("number")\
                    .write()
    
    # Join not yet supported
    if False:
        t850 = Cascade().read(f"date=...,levtype=pl,level=850,number=1/to/50,step={start}/to/{end}/by/12")\
                        .foreach(spectral_transform)\
                        .groupby("step")\
                        .join(climatology)\
                        .foreach(lambda x, y: x - y)\
                        .mean("step")\
                        .threshold("< -2")\
                        .mean("number")\
                        .write()
    
    if False:
        t850 = Cascade().join(t850, climatology).foreach(lambda x, y: x - y).mean("step").threshold("< -2").mean("number").write()

    window_cascades.append(t850)

    # draw graph    
    g = t850.graph()
    pyvis.to_pyvis(g, notebook=True, cdn_resources="remote").show(f"graph_{window}.html")
    total_graph += g

if False:
    # concatenate not yet supported
    Cascade().concatenate(*total_graph).schedule()

if False:
    # some other examples of fluent queries
    take10 = Cascade().orderby(lambda x: x.step).take(10)
    other = Cascade().read()

    # could implement forking (not yet supported)
    take10.foreach(lambda x: x + 1).mean().then(lambda x: x + 1).write()
    take10.mean().join(other).sum().write()



# orderby
# groupby
# take
# where
# sum/max/min/mean
# join (into tuples)

# Cascade::read().joinread()






pyvis.to_pyvis(g, notebook=True, cdn_resources="remote").show("index.html")


# # -----------------------

# Dask notes: 

# cant go into nested functions
# cant do foreach, we have to code it explicitly
# have to wrap every function in a dask.delayed(func)(args) call
# annotations using with: