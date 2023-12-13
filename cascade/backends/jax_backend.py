import jax.numpy as jnp

from .base import BaseBackend


class JaxBackend(BaseBackend):
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
            arrays = jnp.asarray(arrays)
        else:
            arrays = arrays[0]

        return getattr(jnp, name)(arrays, **method_kwargs)

    def mean(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("mean", *arrays, **method_kwargs)

    def std(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("std", *arrays, **method_kwargs)

    def min(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("min", *arrays, **method_kwargs)

    def max(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("max", *arrays, **method_kwargs)

    def sum(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("sum", *arrays, **method_kwargs)

    def prod(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("prod", *arrays, **method_kwargs)

    def var(*arrays, **method_kwargs) -> jnp.ndarray:
        return JaxBackend.multi_arg_function("var", *arrays, **method_kwargs)

    def stack(*arrays, axis: int = 0) -> jnp.ndarray:
        broadcasted = jnp.broadcast_arrays(*arrays)
        return jnp.stack(broadcasted, axis=axis)

    def concat(*arrays, **method_kwargs) -> jnp.ndarray:
        return jnp.concatenate(arrays, **method_kwargs)

    def take(array, indices, *, axis: int, **kwargs):
        if hasattr(indices, "__iter__"):
            return jnp.take(array, jnp.asarray(indices), axis=axis, **kwargs)
        return jnp.take(array, indices, axis=axis, **kwargs)
