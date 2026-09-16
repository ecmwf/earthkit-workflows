import copy 
import itertools
import numpy as np
from typing import Any, Iterable, Optional

def expand(
    datacube: dict[str, Any],
    dims: Optional[list[str]] = None
) -> Iterable[dict[str, Any]]:
    datacube = copy.deepcopy(datacube)
    dims = dims or [x for x in datacube.keys()]
    expansion = {}
    for d in dims:
        coords = datacube.pop(d, None)
        if coords is None:
            continue
        expansion[d] = [coords] if np.ndim(coords) == 0 else list(coords)

    keys = list(expansion.keys())
    its = tuple(expansion.values())
    for vals in itertools.product(*its):
        yield dict(zip(keys, vals))