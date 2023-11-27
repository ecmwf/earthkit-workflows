from cascade.cascade import register_graph
from .products import GRAPHS

for product in GRAPHS:
    register_graph(product.__name__, product)
