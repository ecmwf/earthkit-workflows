# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

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


def _get_name(node: dict[str, Any], index: int) -> str:
    """Extract a branch name from a tree-JSON node's metadata.

    Falls back to an alphabetical label (a, b, c, ...) when the ``name``
    metadata is absent.

    TODO: Once the Rust Qube supports metadata, the ``name`` field will be
    populated automatically.  Until then callers that need named branches
    should set metadata in the tree-JSON dict before calling
    ``expand_as_qube``.
    """
    meta = node.get("metadata", {})
    if "name" in meta:
        name_entry = meta["name"]
        # metadata "name" follows the tree-JSON shape: {"values": [...], ...}
        if isinstance(name_entry, dict) and "values" in name_entry:
            vals = name_entry["values"]
            if vals:
                return str(vals[0])
        # Plain string / scalar metadata (future-proofing)
        return str(name_entry)
    return _convert_num_to_abc(index)


def expand_as_qube(action: Action, qube: Qube, dims: Optional[list[str]] = None) -> Action:
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
    dims: List[str]
        List of dimensions to expand over.

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

    **Important**: When combining qubes using the ``|`` operator, for a parent
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
    >>> pressure = Qube.from_datacube({"step": [6, 12], "param": ["t", "u", "v"], "level": [500, 850, 1000]})
    >>> qube = surface | pressure  # Both have step, so parent has step
    >>> expanded_action = expand_as_qube(action, qube)
    # Result has hierarchical structure:
    # /a - expanded with (step, param)
    # /b - expanded with (step, param, level)

    Drop an axis before expansion:

    >>> qube = Qube.from_datacube({"step": [6, 12], "param": ["t", "q"]})
    >>> qube_without_step = qube.drop(["step"])
    >>> expanded_action = expand_as_qube(action, qube_without_step)
    # Action expanded over param dimension only
    """

    # Serialise the Qube into its tree-JSON representation so we can walk
    # the node hierarchy without needing per-node Python bindings.
    tree: dict[str, Any] = json.loads(qube.to_tree_json())
    expand_dims: list[str] = dims or list(qube.axes().keys())

    leaves: dict[str, Action] = {}

    def _walk(action: Action, node: dict[str, Any], path: str) -> None:
        """Recursively expand *action* by walking the tree-JSON *node*."""
        key: str = node["key"]
        values: list[Any] = node["values"]["values"]
        children: list[dict[str, Any]] = node["children"]

        if key != "root" and key in expand_dims and values:
            action = action.expand(
                (key, values),
                (key, values),
                backend_kwargs={"method": "sel"},
            )

        match len(children):
            case 0:
                assert path not in leaves, f"Duplicate path detected: {path}"
                leaves[path] = action
            case 1:
                _walk(action, children[0], path)
            case _:
                for i, child in enumerate(children):
                    _walk(action, child, f"{path}/{_get_name(child, i)}")

    if not tree.get("children"):
        return action

    _walk(action, tree, "")
    return fluent.merge(**leaves)
