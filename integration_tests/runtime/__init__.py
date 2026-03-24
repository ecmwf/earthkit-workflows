"""Module to be imported from at runtime, ie, the callables definitions"""


def check_numpy_version(expected: str) -> bool:
    import numpy
    if numpy.__version__ != expected:
        raise ValueError(f"version check failure, {numpy.__version__=} != {expected=}")
    return True


def source_42() -> int:
    print("source_42 called")
    return 42

def transform_increment(a: int) -> int:
    print(f"transform_increment called with input {a}")
    return a+1

def product_add(a: int, b: int) -> int:
    print(f"product_add called with input ({a}, {b})")
    return a+b

def sink_file(data, fname: str) -> None:
    print(f"sink_file called with {fname=} and {data=}")
    import pathlib
    pathlib.Path(fname).write_text(str(data))
