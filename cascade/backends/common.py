def trivial(arg):
    return arg


def method(array, name: str, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)
