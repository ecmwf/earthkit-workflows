import pytest

try:
    import cascade.graph as cgraph
except ImportError:
    cgraph = None


def test_cgraph_exists():
    assert cgraph is not None


@pytest.mark.parametrize("name", ["Node", "Source", "Processor", "Sink"])
def test_class_exists(name):
    assert hasattr(cgraph, name)
    obj = getattr(cgraph, name)
    assert isinstance(obj, type)


def test_class_hierarchy():
    assert issubclass(cgraph.Source, cgraph.Node)
    assert issubclass(cgraph.Processor, cgraph.Node)
    assert issubclass(cgraph.Sink, cgraph.Node)
