from typing import Any
import torch.fx as fx
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path


def is_type(obj, type: type):
    """Aborts if obj is not of type type."""
    assert isinstance(obj, type), f"Object {obj} is not of type {type}."


def static_single_assignment(n: Sequence[fx.Node]):
    """Aborts if any node appears multiple times in n"""
    seen: set[str] = set()
    for node in n:
        assert node.name not in seen, f"Node {node.name} appears multiple times."
        seen.add(node.name)


def topological_order(n: Sequence[fx.Node]):
    """Aborts if nodes are not in topological order. Requires static single assignment."""
    static_single_assignment(n)

    params: set[str] = set()
    for node in n:
        assert node.name not in params, f"Node {node.name} appears before its dependencies."
        params |= {inp.name for inp in node.all_input_nodes}


def file_extension(p: Path, ext: str):
    """Aborts if path p does not have extension ext."""
    assert p.suffix == ext, f"File {p} does not have extension {ext}."


def subset(a: Iterable[Any], b: Iterable[Any]):
    """Aborts if a is not a subset of b.."""
    assert set(a).issubset(set(b)), "a is not a subset of b."


def disjoint(a: Iterable[Any], b: Iterable[Any]):
    """Aborts if a and b are not disjoint"""
    assert set(a).isdisjoint(set(b)), "a and b are not disjoint."


def disjoint_and_complete(u: Sequence[fx.Node], p: Sequence[Sequence[fx.Node]]):
    """AAborts if partitions p are not disjoint and complete over (u)niverse."""

    num_nodes = sum(len(part) for part in p)
    all_nodes = set().union(*(set(part) for part in p))

    assert num_nodes == len(u), "Partitions do not cover all nodes."
    assert len(all_nodes) == num_nodes, "Partitions are not disjoint."
