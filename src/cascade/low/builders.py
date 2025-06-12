# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import inspect
import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Iterator, Type, cast

import pyrsistent
from typing_extensions import Self

from cascade.low.core import (
    DatasetId,
    JobInstance,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
)
from cascade.low.func import Either
from earthkit.workflows.graph import Node


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
            output_schema=[(Node.DEFAULT_OUTPUT, type2str(sig.return_annotation))],
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
            output_schema=[(Node.DEFAULT_OUTPUT, output_class)],
        )
        return cls(definition=definition, static_input_kw={}, static_input_ps={})

    def with_values(self, *args, **kwargs) -> Self:
        new_kwargs = {**self.static_input_kw, **kwargs}
        ps_args = {str(k): v for k, v in dict(enumerate(args)).items()}
        new_args = {**self.static_input_ps, **ps_args}
        return self.model_copy(
            update={"static_input_kw": new_kwargs, "static_input_ps": new_args}
        )


@dataclass
class JobBuilder:
    nodes: pyrsistent.PMap = field(default_factory=lambda: pyrsistent.m())
    edges: pyrsistent.PVector = field(default_factory=lambda: pyrsistent.v())

    def with_node(self, name: str, task: TaskInstance) -> Self:
        return replace(self, nodes=self.nodes.set(name, task))

    def with_edge(
        self, source: str, sink: str, into: str | int, frum: str = Node.DEFAULT_OUTPUT
    ) -> Self:
        new_edge = Task2TaskEdge(
            source=DatasetId(source, frum),
            sink_task=sink,
            sink_input_kw=into if isinstance(into, str) else None,
            sink_input_ps=into if isinstance(into, int) else None,
        )
        return replace(self, edges=self.edges.append(new_edge))

    def build(self) -> Either[JobInstance, list[str]]:
        # TODO replace `_isinstance` with a smarter check for self-reg types, reuse fiab/type_system
        skipped = {
            "latitude",
            "longitude",
            "latlonArea",
            "Optional[marsParam]",
            "marsParamList",
            "grib",
        }
        _isinstance = lambda v, t: t == "Any" or t in skipped or isinstance(v, eval(t))

        # static input types
        static_kw_errors: Iterable[str] = (
            f"invalid static input for {task}: {k} needs {instance.definition.input_schema[k]}, got {type(v)}"
            for task, instance in self.nodes.items()
            for k, v in instance.static_input_kw.items()
            if not _isinstance(v, instance.definition.input_schema[k])
        )

        # edge correctness
        def get_edge_errors(edge: Task2TaskEdge) -> Iterator[str]:
            source_task = self.nodes.get(edge.source.task, None)
            output_param = None
            if not source_task:
                yield f"edge pointing from non-existent task {edge.source}"
            else:
                for key, schema in source_task.definition.output_schema:
                    if key == edge.source.output:
                        output_param = schema
                if not output_param:
                    yield f"edge pointing from non-existent param {edge.source.output}"
            sink_task = self.nodes.get(edge.sink_task, None)
            if not sink_task:
                yield f"edge pointing to non-existent task {edge.sink_task}"
            else:
                if edge.sink_input_kw is None:
                    return
                input_param = sink_task.definition.input_schema.get(
                    edge.sink_input_kw, None
                )
                if not input_param:
                    yield f"edge pointing to non-existent param {edge.sink_input_kw}"
            if not output_param or not input_param:
                return
            # TODO replace `issubclass` with a smarter check for self-reg types
            legits = {("grib.earthkit", "grib.mir"), ("grib.mir", "grib.earthkit")}
            _issubclass = (
                lambda t1, t2: t2 == "Any"
                or t1 == t2
                or (t1, t2) in legits
                or issubclass(eval(t1), eval(t2))
            )
            if not _issubclass(output_param, input_param):
                yield f"edge connects two incompatible nodes: {edge}"

        edge_errors: Iterable[str] = (
            error for edge in self.edges for error in get_edge_errors(edge)
        )

        # all inputs present
        # TODO

        errors = list(itertools.chain(static_kw_errors, edge_errors))
        if errors:
            return Either.error(errors)
        else:
            return Either.ok(
                JobInstance(
                    tasks=cast(dict[str, TaskInstance], pyrsistent.thaw(self.nodes)),
                    edges=pyrsistent.thaw(self.edges),
                )
            )
