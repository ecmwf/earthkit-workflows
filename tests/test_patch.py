import jax.numpy as jnp
import pytest

from meteokit import extreme

from cascade.patch import PatchModule


def test_patch_extreme():
    a = jnp.ones((2, 3))
    with PatchModule(extreme, "numpy", jnp):
        with pytest.raises(Exception):
            extreme.efi(a, a, 0.0001)
    extreme.efi(a, a, 0.0001)
