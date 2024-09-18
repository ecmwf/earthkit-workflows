import inspect
from dataclasses import dataclass, field, replace
from typing import Callable, Type, cast

import pyrsistent
from typing_extensions import Self

from cascade.graph import Node
from cascade.v2.core import JobInstance, Task2TaskEdge, TaskDefinition, TaskInstance


class TaskBuilder(TaskInstance):
    @classmethod
    def from_callable(cls, f: Callable, environment: list[str] | None = None) -> Self:
        def type2str(t: str | Type) -> str:
            type_name: str = t if isinstance(t, str) else t.__name__
            return "Any" if type_name == "_empty" else type_name

        sig = inspect.signature(f)
        input_schema = {
            p.name: type2str(p.annotation)
            for p in sig.parameters.values()
            if p.kind
            in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        }
        static_input_kw = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.kind
            in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            and p.default != inspect.Parameter.empty
        }

        definition = TaskDefinition(
            entrypoint="",
            func=TaskDefinition.func_enc(f),
            environment=environment if environment else [],
            input_schema=input_schema,
            output_schema={Node.DEFAULT_OUTPUT: type2str(sig.return_annotation)},
        )
        return cls(
            definition=definition, static_input_kw=static_input_kw, static_input_ps={}
        )

    @classmethod
    def from_entrypoint(
        cls,
        entrypoint: str,
        input_schema: dict[str, str],
        output_class: str,
        environment: list[str] | None = None,
    ) -> Self:
        # NOTE this isnt really practical -- for entrypoint-based tasks, it makes more sense to have a util
        # class that derives the signature/env dynamically with all imports in place, creates
        # TaskInstance like `from_callable` except for swapping the entrypoint, and serializes
        definition = TaskDefinition(
            entrypoint=entrypoint,
            func=None,
            environment=environment if environment else [],
            input_schema=input_schema,
            output_schema={Node.DEFAULT_OUTPUT: output_class},
        )
        return cls(definition=definition, static_input_kw={}, static_input_ps={})

    def with_values(self, *args, **kwargs) -> Self:
        new_kwargs = {**self.static_input_kw, **kwargs}
        new_args = {**self.static_input_ps, **dict(enumerate(args))}
        return self.model_copy(
            update={"static_input_kw": new_kwargs, "static_input_ps": new_args}
        )


@dataclass
class JobBuilder:
    nodes: pyrsistent.PMap = field(default_factory=lambda: pyrsistent.m())
    edges: pyrsistent.PVector = field(default_factory=lambda: pyrsistent.v())

    def with_node(self, name: str, task: TaskInstance) -> Self:
        return replace(self, nodes=self.nodes.set(name, task))

    def with_edge(self, source: str, sink: str, into: str | int) -> Self:
        new_edge = Task2TaskEdge(
            source_task=source,
            source_output=Node.DEFAULT_OUTPUT,
            sink_task=sink,
            sink_input_kw=into if isinstance(into, str) else None,
            sink_input_ps=into if isinstance(into, int) else None,
        )
        return replace(self, edges=self.edges.append(new_edge))

    def build(self) -> JobInstance:
        return JobInstance(
            tasks=cast(dict[str, TaskInstance], pyrsistent.thaw(self.nodes)),
            edges=pyrsistent.thaw(self.edges),
        )
