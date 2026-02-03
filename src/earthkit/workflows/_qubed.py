# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qubed import Qube

    from earthkit.workflows.fluent import Action

def _convert_num_to_abc(num: int) -> str:
    """Convert a number to its corresponding alphabetical representation.
    0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa', etc.
    """
    result = ""
    while True:
        num, remainder = divmod(num, 26)
        result = chr(97 + remainder) + result
        if num == 0:
            break
        num -= 1
    return result

def get_name(child: "Qube", index: int) -> str:
    if "name" in child.metadata:
        name_meta = child.metadata["name"]
        return str(np.unique_values(name_meta).flatten()[0])
    return _convert_num_to_abc(index)

def _select(x: Any, key: str, val: Any):
    return x.sel(**{key: val})

def expand_as_qube(action: "Action", qube: "Qube") -> "Action":
    """Expand an action according to a qube structure.

    This function recursively expands an action across all dimensions defined
    in the qube. For qubes with a single child, it expands sequentially through
    the dimensions. For qubes with multiple children, it splits the action into
    separate branches, each named according to the child's metadata (if present)
    or using alphabetical naming as a fallback.

    Parameters
    ----------
    action : Action
        The workflow action to expand across the qube's dimensions.
    qube : Qube
        The qube defining the structure and dimensions for expansion.

    Returns
    -------
    Action
        The expanded action with all dimensions applied. The action will have
        a hierarchical structure if the qube has multiple children.

    Notes
    -----
    The expansion algorithm works as follows:

    1. If the qube has no children, the action is returned unchanged
    2. The action is expanded along each dimension in the qube hierarchy
    3. If multiple children exist, the action is split into branches
    4. Each branch is recursively expanded with its child's dimensions
    5. Child branches are named using metadata or alphabetical fallback

    Examples
    --------
    Simple single-dimension expansion:

    >>> from qubed import Qube
    >>> qube = Qube.from_datacube({"step": [6, 12, 18]})
    >>> expanded_action = expand_as_qube(action, qube)
    # Action is now expanded over step=[6, 12, 18]

    Hierarchical expansion with surface and pressure levels:

    >>> # Create a hierarchical qube structure:
    >>> # root, step=6/12
    >>> # ├── param=2t/10u/10v (surface)
    >>> # └── param=t/u/v, level=500/850/1000 (pressure)
    >>> surface = Qube.from_datacube({"param": ["2t", "10u", "10v"]})
    >>> surface.add_metadata({"name": "surface"})
    >>> pressure = Qube.from_datacube({"param": ["t", "u", "v"], "level": [500, 850, 1000]})
    >>> pressure.add_metadata({"name": "pressure"})
    >>> parent = Qube.from_datacube({"step": [6, 12]})
    >>> qube = parent | (surface | pressure)
    >>> expanded_action = expand_as_qube(action, qube)
    # Action is expanded over step, then split into /surface and /pressure
    # branches, each with their respective param and level dimensions

    Drop an axis before expansion:

    >>> qube = Qube.from_datacube({"step": [6, 12], "param": ["t", "q"]})
    >>> qube_without_step = qube.remove_by_key("step")
    >>> expanded_action = expand_as_qube(action, qube_without_step)
    # Action expanded over param dimension only
    """

    try:
        from qubed import Qube
    except ImportError:
        raise ImportError("The 'qubed' package is required for this function. Please install it.")
    
    if not isinstance(qube, Qube):
        raise TypeError(f"'qube' must be an instance of Qube, got {type(qube)}")
    
    from earthkit.workflows.fluent import Payload

    def expand_fn(action: "Action", qube: "Qube", path: str) -> "Action":
        """Recursively expand the action based on the qube structure."""
        if not qube.key == "root":  # Skip the root key
            # Expand along the current qube's key and values
            action = action.expand((qube.key, list(qube.values)), (qube.key, list(qube.values)), path=path)

        num_children = len(qube.children)
        if num_children == 0:  # Base case: no more children to expand
            return action

        if num_children == 1:  # In the case of one child, no need to split, just continue expanding
            return expand_fn(action, qube.children[0], path)

        action = action.split(
            {
                f"{path}/{get_name(child, i)}": Payload(
                    _select, kwargs={"key": child.key, "val": list(child.values)}
                )
                for i, child in enumerate(qube.children)
            }
        )
        for i, child in enumerate(qube.children):
            sub_path = f"{path}/{get_name(child, i)}"
            action = expand_fn(action, child, sub_path)

        return action

    if not qube.children:
        return action

    return expand_fn(action, qube, "")
