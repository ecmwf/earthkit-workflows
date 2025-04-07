# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from typing_extensions import Self


class Output:
    """Helper class to refer to node outputs"""

    parent: "Node"
    name: str

    def __init__(self, parent: "Node", name: str):
        self.parent = parent
        self.name = name

    def serialise(self) -> str | tuple[str, str]:
        """Convert the reference to a serialisable type"""
        if self.name == Node.DEFAULT_OUTPUT:
            return self.parent.name
        return self.parent.name, self.name

    def __repr__(self) -> str:
        if self.name == Node.DEFAULT_OUTPUT:
            return f"<Default output of {self.parent!r}>"
        return f"<Output {self.name!r} of {self.parent!r}>"

    def __str__(self) -> str:
        if self.name == Node.DEFAULT_OUTPUT:
            return self.parent.name
        return f"{self.parent.name}.{self.name}"


class Node:
    """Base class for graph nodes

    A node is defined with references to its actual inputs, and can have either
    one default output, or any number of named outputs (including zero). Named
    outputs can be accessed by attribute lookup (e.g.: ``node.output1``), and
    any output, including the default output, can be obtained with `get_output`.

    A node has the following attributes:
    - `name`, its name (should be unique within a graph)
    - `inputs`, a mapping between input names and outputs of other nodes
    - `outputs`, a list of outputs
    - `payload`, a generic payload

    Parameters
    ----------
    name: str
        Node name
    outputs: list[str] | None
        List of outputs. If None or not set, assumes the node only has one
        default output
    payload: Any
        Node payload
    **inputs: Node | Output
        Node outputs to use as inputs. If a `Node` object is passed, the input
        will be connected to its default output.
    """

    DEFAULT_OUTPUT = "0"

    name: str
    inputs: dict[str, Output]
    outputs: list[str]
    payload: Any

    def __init__(
        self,
        name: str,
        outputs: list[str] | None = None,
        payload: Any = None,
        **kwargs: "Node | Output",  # NOTE can't declare Self due to children. Fix hiearchy instead
    ):
        self.name = name
        self.outputs = [Node.DEFAULT_OUTPUT] if outputs is None else outputs
        self.payload = payload
        self.inputs = {
            iname: (inp if isinstance(inp, Output) else inp.get_output())
            for iname, inp in kwargs.items()
        }

    def __getattr__(self, name: str) -> Output:
        return self.get_output(name)

    def get_output(self, name: str | None = None) -> Output:
        """Get an output from the node

        If ``name`` is ``None``, the node is expected to have a default output,
        which is returned. Otherwise the output with the given name is returned.
        Raises `AttributeError` if the output does not exist.
        """
        if name is None:
            name = Node.DEFAULT_OUTPUT
        if name in self.outputs:
            return self._make_output(name)
        if name == Node.DEFAULT_OUTPUT:
            raise AttributeError(f"Node {self.name} has no default output")
        raise AttributeError(name)

    def _make_output(self, name: str) -> Output:
        return Output(self, name)

    def serialise(self) -> dict[str, Any]:
        """Convert the node to a serialisable value

        If the payload object has a ``serialise`` method, it is called without
        arguments to get its serialised form, otherwise the payload is assumed
        to be serialisable as is.
        """
        res: dict[str, Any] = {}
        res["outputs"] = self.outputs.copy()
        res["inputs"] = {name: src.serialise() for name, src in self.inputs.items()}
        if self.payload is not None:
            if hasattr(self.payload, "serialise"):
                res["payload"] = self.payload.serialise()
            else:
                res["payload"] = self.payload
        return res

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name!r} at {id(self):#x}>"

    def is_source(self) -> bool:
        """Check whether the node is a source (i.e. has no inputs)"""
        return not self.inputs

    def is_processor(self) -> bool:
        """Check whether the node is a processor (i.e. has both inputs and outputs)"""
        return (not self.is_sink()) and (not self.is_source())

    def is_sink(self) -> bool:
        """Check whether the node is a sink (i.e. has no outputs)"""
        return not self.outputs

    def copy(self) -> Self:
        """Shallow copy of the node (the payload is not copied)"""
        return self.__class__(
            self.name, self.outputs.copy(), self.payload, **self.inputs
        )
