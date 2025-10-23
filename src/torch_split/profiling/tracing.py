"""Adds OpenTelemetry instrumentation to PyTorch models to learn the distribution of time spent and size of objects at each layer during inference."""

from torch_split.logging import get_logger

logger = get_logger(__name__)
