import copy
import itertools
from typing import Any, Iterable, Iterator, Optional

import numpy as np
from qubed import Qube  # type: ignore


def expand_datacube(datacube: dict[str, Any], dims: Optional[list[str]] = None) -> Iterable[dict[str, Any]]:
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
        yield dict(zip(keys, vals), **{k: v for k, v in datacube.items() if k not in expansion})


def qube_to_datacubes(qube: Qube, expand: bool = False) -> Iterator[dict[str, Any]]:
    for cube in qube.to_datacubes():
        cube.pop("root", None)
        if not expand:
            if len(cube) == 0:
                continue
            yield cube
        else:
            for expanded in expand_datacube(cube):
                if len(expanded) == 0:
                    continue
                yield expanded


def num_leaves(qube: Qube) -> int:
    """Return the number of leaves in a qube"""
    return len(list(qube_to_datacubes(qube, expand=True)))
