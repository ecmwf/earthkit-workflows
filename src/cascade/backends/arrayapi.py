import array_api_compat


def _xp_multi_args(name: str, *args, **kwargs):
    xp = array_api_compat.array_namespace(*args)
    if len(args) > 1:
        kwargs["axis"] = 0
    else:
        args = args[0]
    return getattr(xp, name)(xp.asarray(args), **kwargs)


class ArrayAPIBackend:
    def mean(*args, **kwargs):
        return _xp_multi_args("mean", *args, **kwargs)

    def std(*args, **kwargs):
        return _xp_multi_args("std", *args, **kwargs)

    def max(*args, **kwargs):
        return _xp_multi_args("max", *args, **kwargs)

    def min(*args, **kwargs):
        return _xp_multi_args("min", *args, **kwargs)

    def sum(*args, **kwargs):
        return _xp_multi_args("sum", *args, **kwargs)

    def prod(*args, **kwargs):
        return _xp_multi_args("prod", *args, **kwargs)

    def var(*args, **kwargs):
        return _xp_multi_args("var", *args, **kwargs)

    def stack(*args, axis: int | None = None, **kwargs):
        xp = array_api_compat.array_namespace(*args)
        broadcasted = xp.broadcast_arrays(*args)
        return xp.stack(broadcasted, axis=axis)

    def concat(*args, **kwargs):
        xp = array_api_compat.array_namespace(*args)
        return xp.concat(args, **kwargs)

    def add(*args):
        return args[0] + args[1]

    def subtract(*args):
        return args[0] - args[1]

    def multiply(*args):
        return args[0] * args[1]

    def divide(*args):
        return args[0] / args[1]

    def pow(*args):
        return args[0] ** args[1]

    def take(array, indices, *, axis: int, **kwargs):
        xp = array_api_compat.array_namespace(array)

        if hasattr(indices, "__iter__"):
            return xp.take(array, indices, axis=axis, **kwargs)
        ret = xp.take(array, [indices], axis=axis, **kwargs)
        return xp.squeeze(ret, axis=axis)
