GRAPHS: dict = {}


# TODO: Determine what Cascade methods should look like in terms of
# supporting different executors and schedulers. Do we need it?
class Cascade:
    def graph(product, *args, **kwargs):
        if product not in GRAPHS:
            raise Exception(f"No graph for '{product}' registered")
        return GRAPHS[product](*args, **kwargs)


def register_graph(name: str, func):
    assert name not in GRAPHS
    GRAPHS[name] = func
