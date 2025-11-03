from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import msgpack  # type: ignore
from pydantic import BaseModel, Field, model_validator

from src.torch_split.logging.logging import get_logger

logger = get_logger(__name__)


class Context(BaseModel):
    trace_id: str
    span_id: str
    trace_state: str


class Status(BaseModel):
    status_code: str


class ResourceAttributes(BaseModel):
    telemetry_sdk_language: str = Field(alias="telemetry.sdk.language")
    telemetry_sdk_name: str = Field(alias="telemetry.sdk.name")
    telemetry_sdk_version: str = Field(alias="telemetry.sdk.version")
    service_name: str = Field(alias="service.name")

    class Config:
        populate_by_name = True  # allows using either alias or pythonic name


class Resource(BaseModel):
    attributes: ResourceAttributes
    schema_url: str


class SpanAttributes(BaseModel):
    group_id: Optional[str] = Field(alias="group.id", default=None)
    layer_type: str = Field(alias="layer.type")
    layer_name: str = Field(alias="layer.name")
    layer_class: str = Field(alias="layer.class")
    layer_source: str = Field(alias="layer.source")
    output_size_bytes: int = Field(alias="output.size.bytes")


class Span(BaseModel):
    name: str
    context: Context
    kind: str
    parent_id: Optional[str]
    start_time: str
    end_time: str
    status: Status
    attributes: SpanAttributes
    events: List[Any]
    links: List[Any]
    resource: Resource

    @model_validator(mode="after")
    def check_root_invariant(self):
        """only root spans have the group_id attributes"""
        if self.parent_id is None and self.attributes.group_id is None:
            raise ValueError("If parent_id is None, group_id  must be provided")
        elif self.parent_id is not None and self.attributes.group_id is not None:
            raise ValueError("If parent_id is not None, group_id must be None")
        return self


@dataclass
class SpanTreeNode:
    span: Span
    children: List["SpanTreeNode"]

    @property
    def layer_identifier(self) -> str:
        return self.span.attributes.layer_source.replace(".", "_")

    def pretty_print(self, level: int = 0, start_time: Optional[float] = None):
        # get start time, and every subsequent level is says start time in
        # ms since start_time
        now = datetime.fromisoformat(self.span.start_time.replace("Z", "+00:00")).timestamp()

        if level == 0:
            start_time = now
            time_elapsed_ms = 0.0
        else:
            assert start_time is not None
            time_elapsed_ms = (now - start_time) * 1000.0

        name_padding = 20 - level
        name_str = f"{self.span.name:<{name_padding}}"
        layer_str = f"{self.span.attributes.layer_name:<20}"

        id_str = f"(ID: {self.span.context.span_id}) {self.span.attributes.output_size_bytes}"
        time_str = f"({time_elapsed_ms:8.2f} ms)"
        print(f"{' ' * level}- {name_str} {layer_str} {id_str:<24} {time_str}")

        for child in self.children:
            child.pretty_print(level + 1, start_time)


def spans_from_file(file_path: str | Path) -> List[SpanTreeNode] | None:
    """Load spans from a msgpack binary file containing a list of spans"""

    file_path = file_path if isinstance(file_path, Path) else Path(file_path)
    assert file_path.exists(), f"File {file_path} does not exist"

    spans: list[Span] = []
    with open(file_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        for obj in unpacker:
            try:
                spans.append(Span(**obj))
            except Exception as e:
                print(f"Error parsing span: {e}")
                return None

    # build span tree
    span_map: dict[str, SpanTreeNode] = {s.context.span_id: SpanTreeNode(s, []) for s in spans}
    roots: List[SpanTreeNode] = []

    for span in spans:
        if span.parent_id and span.parent_id in span_map:
            parent = span_map[span.parent_id]
            parent.children.append(span_map[span.context.span_id])
        else:
            roots.append(span_map[span.context.span_id])

    logger.info(f"Loaded {len(spans)} spans from {file_path}, {len(roots)} roots")
    return roots
