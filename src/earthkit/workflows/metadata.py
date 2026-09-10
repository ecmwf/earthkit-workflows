from typing import Optional

from pydantic import Field

from cascade.low.func import CascadeBaseModel


class Requirements(CascadeBaseModel):
    environment: list[str] = Field(
        description="pip-installable packages, as required by entrypoint/func. Version pins supported", default_factory=list
    )
    needs_gpu: Optional[bool] = Field(description="whether the task requires a GPU to run", default=None)


class Artifacts(CascadeBaseModel):
    artifact_urls: dict[str, str] = Field(description="mapping of artifact names to their URLs", default_factory=dict)


class BuilderMetadata(CascadeBaseModel):
    blockId: Optional[str] = Field(description="unique identifier for the block that generated this node", default=None)


class NodeMetadata(CascadeBaseModel):
    requirements: Requirements = Field(description="requirements for the node", default_factory=Requirements)
    artifacts: Artifacts = Field(description="artifacts produced by the node", default_factory=Artifacts)
    builder: BuilderMetadata = Field(description="metadata about the builder that created this node", default_factory=BuilderMetadata)


def update_requirements(mInto: Requirements, mFrom: Requirements) -> None:
    mInto.environment = list(set(mFrom.environment + mInto.environment))
    mInto.needs_gpu = mFrom.needs_gpu or mInto.needs_gpu


def update_artifacts(mInto: Artifacts, mFrom: Artifacts) -> None:
    mInto.artifact_urls.update(mFrom.artifact_urls)


def update_builder_metadata(mInto: BuilderMetadata, mFrom: BuilderMetadata) -> None:
    mInto.blockId = mFrom.blockId or mInto.blockId


def update_node_metadata(mInto: NodeMetadata, mFrom: NodeMetadata) -> None:
    update_requirements(mInto.requirements, mFrom.requirements)
    update_artifacts(mInto.artifacts, mFrom.artifacts)
    update_builder_metadata(mInto.builder, mFrom.builder)
