from cascade import backends


class CustomBackend:
    def write(arg):
        return arg

    def sum(arg1, arg2):
        return arg1 + arg2


def test_custom():
    backends.register(str, CustomBackend)

    # For graph merging, custom functions must be the
    # unique e.g. attribute must be the same each access
    func = backends.write
    func1 = backends.write
    assert func == func1

    assert backends.write("Helloworld!") == backends.sum("Hello", "world!")
