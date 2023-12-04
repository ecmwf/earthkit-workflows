import array_api_compat


def trivial(arg):
    return arg


def stack(*args, axis: int = 0):
    xp = array_api_compat.array_namespace(*args)
    return xp.stack(args, axis)


def get_item(arg, index: int):
    return arg[index]
