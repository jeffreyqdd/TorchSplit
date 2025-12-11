from collections.abc import Generator
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms  # type: ignore
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
)

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from torch_split.runtime import SwitchboardRuntime

# Set up console exporter for tracing
resource = Resource.create({"service.name": "torchsplit-runtime"})
# console_exporter = ConsoleSpanExporter()
jaeger_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
trace_provider = TracerProvider(resource=resource)
# trace_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
trace_provider.add_span_processor(SimpleSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(trace_provider)

x = SwitchboardRuntime(Path("/dev/shm/switchboard.tspartd"))

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# original_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# original_model.eval()

# # # basic input
img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(128)]
texts = [f"a plain white square {9}" for i in range(128)]
enc = processor(
    images=img,
    text=texts,
    padding="max_length",  # type: ignore
    max_length=32,  # type: ignore
    return_tensors="pt",  # type: ignore
)  # type: ignore
pixel_values = enc["pixel_values"].to("cuda")
input_ids = enc["input_ids"].to("cuda")
attention_mask = enc["attention_mask"].to("cuda")

for i in range(1000):
    result = x.interpret(
        l_pixel_values_=pixel_values,
        l_input_ids_=input_ids,
        l_attention_mask_=attention_mask,
    )
    print(result["C"])
# original_output = original_model(
#     pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
# ).logits_per_image

# print("original", original_output)
# print("split", result.keys())
# print("split output", result["C"])

# # # Calculate error metrics
# # abs_diff = torch.abs(original_output.to("cuda") - result)
# # rel_diff = abs_diff / torch.abs(original_output.to("cuda"))
# # print(f"Absolute difference: {abs_diff.item():.6f}")
# # print(f"Relative difference: {rel_diff.item():.4%}")
