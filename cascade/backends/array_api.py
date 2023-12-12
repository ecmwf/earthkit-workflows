import array_api_compat
import functools

from .base import BaseBackend


class ArrayApiBackend(BaseBackend):
    def multi_arg_function(name: str, *arrays, **method_kwargs):
        """
        Apply named function in array module on arrays. If arrays
        only consists of single array then function is applied
        along an axis (default is axis 0) of this single array.

        Parameters
        ----------
        name: str, name of function to apply
        arrays: ArrayLike, list of arrays to apply function on

        Return
        ------
        ArrayLike, result of applying named function
        """
        if len(arrays) > 1:
            method_kwargs["axis"] = 0
        else:
            arrays = arrays[0]
            # method_kwargs.setdefault("axis", 0)

        xp = array_api_compat.array_namespace(*arrays)
        return getattr(xp, name)(arrays, **method_kwargs)

    def single_arg_function(name: str, *args, **method_kwargs):
        """
        Apply named function in array module on array

        Parameters
        ----------
        name: str, name of function to apply
        array: ArrayLike

        Return
        ------
        ArrayLike, result of applying named function
        """
        xp = array_api_compat.array_namespace(args[0])
        return getattr(xp, name)(*args, **method_kwargs)


for func in ["mean", "std", "min", "max", "sum", "prod", "var", "stack", "concat"]:
    setattr(
        ArrayApiBackend,
        func,
        functools.partial(ArrayApiBackend.multi_arg_function, func),
    )

for func in ["take"]:
    setattr(
        ArrayApiBackend,
        func,
        functools.partial(ArrayApiBackend.single_arg_function, func),
    )
