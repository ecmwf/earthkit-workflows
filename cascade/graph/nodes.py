from typing import Any


class Node:
    DEFAULT_OUTPUT = "__default__"

    class Output:
        parent: "Node"
        name: str

        def __init__(self, parent: "Node", name: str):
            self.parent = parent
            self.name = name

        def serialise(self) -> str | tuple[str, str]:
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

    name: str
    inputs: dict[str, "Node.Output"]
    outputs: list[str]
    payload: Any

    def __init__(
        self,
        name: str,
        outputs: list[str] | None = None,
        payload: Any = None,
        **kwargs: "Node | Node.Output",
    ):
        self.name = name
        self.outputs = [Node.DEFAULT_OUTPUT] if outputs is None else outputs
        self.payload = payload
        self.inputs = {
            iname: (inp if isinstance(inp, Node.Output) else inp.get_output())
            for iname, inp in kwargs.items()
        }

    def __getattr__(self, name: str) -> Output:
        return self.get_output(name)

    def get_output(self, name: str | None = None) -> Output:
        if name is None:
            name = Node.DEFAULT_OUTPUT
        if name in self.outputs:
            return self._make_output(name)
        if name == Node.DEFAULT_OUTPUT:
            raise AttributeError(f"Node {self.name} has no default output")
        raise AttributeError(name)

    def _make_output(self, name: str) -> Output:
        return Node.Output(self, name)

    def serialise(self) -> dict:
        res = {}
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
        return not self.inputs

    def is_processor(self) -> bool:
        return bool(self.inputs) and bool(self.outputs)

    def is_sink(self) -> bool:
        return not self.outputs

    def copy(self) -> "Node":
        return Node(self.name, self.outputs.copy(), self.payload, **self.inputs)


class Source(Node):
    def __init__(
        self, name: str, outputs: list[str] | None = None, payload: Any = None
    ):
        super().__init__(name, outputs, payload)

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, Node) and instance.is_source()

    def copy(self) -> "Source":
        return Source(self.name, self.outputs.copy(), self.payload)


class Processor(Node):
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, Node) and instance.is_processor()

    def copy(self) -> "Processor":
        return Processor(self.name, self.outputs.copy(), self.payload, **self.inputs)


class Sink(Node):
    def __init__(self, name: str, payload: Any = None, **kwargs: Node | Node.Output):
        super().__init__(name, [], payload, **kwargs)

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, Node) and instance.is_sink()

    def copy(self) -> "Sink":
        return Sink(self.name, self.payload, **self.inputs)
