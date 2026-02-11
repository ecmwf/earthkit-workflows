# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import TYPE_CHECKING

import numpy as np

from earthkit.workflows import fluent

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

def expand_as_qube(action: "Action", qube: "Qube") -> "Action":
    """Expand an action according to a qube structure.

    This function recursively traverses the qube hierarchy and expands the action
    across all dimensions. The qube represents the underlying data's dimensions.
    When the qube has multiple children, the function creates separate branches
    in the resulting action, merging them into a hierarchical DataTree structure.

    Parameters
    ----------
    action : Action
        The workflow action to expand across the qube's dimensions.
    qube : Qube
        The qube defining the structure and dimensions for expansion.
        Qubes represent the underlying data's dimensions.

    Returns
    -------
    Action
        The expanded action with all dimensions applied. The action will have
        a hierarchical DataTree structure if the qube has multiple children.

    Notes
    -----
    The expansion algorithm works as follows:

    1. If the qube has no children, the action is returned unchanged
    2. For each qube node, expand the action along its dimension (key and values)
    3. If a qube has a single child, continue expanding recursively
    4. If a qube has multiple children:
       - Recursively expand each child, storing results as leaves
       - Merge all leaf actions into a single hierarchical structure
    5. Child branches are named using the qube's "name" metadata or alphabetical
       fallback (a, b, c, ...)

    **Important**: When combining qubes using the `|` operator, for a parent
    qube to have a dimension, BOTH children must share that dimension.

    Examples
    --------
    Simple single-dimension expansion:

    >>> from qubed import Qube
    >>> qube = Qube.from_datacube({"step": [6, 12, 18]})
    >>> expanded_action = expand_as_qube(action, qube)
    # Action is now expanded over step=[6, 12, 18]

    Hierarchical expansion with surface and pressure levels:

    >>> # Create a hierarchical qube where both children have step dimension
    >>> # Structure:
    >>> # root, step=6/12
    >>> # ├── name=surface, param=2t/10u/10v
    >>> # └── name=pressure, param=t/u/v, level=500/850/1000
    >>> surface = Qube.from_datacube({"step": [6, 12], "param": ["2t", "10u", "10v"]})
    >>> surface.add_metadata({"name": "surface"})
    >>> pressure = Qube.from_datacube({"step": [6, 12], "param": ["t", "u", "v"], "level": [500, 850, 1000]})
    >>> pressure.add_metadata({"name": "pressure"})
    >>> qube = surface | pressure  # Both have step, so parent has step
    >>> expanded_action = expand_as_qube(action, qube)
    # Result has hierarchical structure:
    # /surface - expanded with (step, param)
    # /pressure - expanded with (step, param, level)

    Drop an axis before expansion:

    >>> qube = Qube.from_datacube({"step": [6, 12], "param": ["t", "q"]})
    >>> qube_without_step = qube.remove_by_key("step")
    >>> expanded_action = expand_as_qube(action, qube_without_step)
    # Action expanded over param dimension only
    """

    leaves: dict[str, Action] = {}

    def expand_fn(action: "Action", qube: "Qube", path: str) -> "Action":
        """Recursively expand the action based on the qube structure."""
        if not qube.key == "root":  # Skip the root key
            # Expand along the current qube's key and values
            action = action.expand((qube.key, list(qube.values)), (qube.key, list(qube.values)))

        match len(qube.children):
            case 0:
                assert path not in leaves, f"Duplicate path detected: {path}"
                leaves[path] = action
            case 1:  # In the case of one child, no need to split, just continue expanding
                expand_fn(action, qube.children[0], path)
            case _:  # Multiple children, need to split into branches
                for i, child in enumerate(qube.children):
                    expand_fn(action, child, f"{path}/{get_name(child, i)}")
                
        return fluent.merge(**leaves)
    
    if not qube.children:
        return action
    return expand_fn(action, qube, "")
