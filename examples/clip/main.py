from collections.abc import Generator
from pathlib import Path
from typing import Any, NoReturn

import torch
import torch.nn as nn
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from PIL import Image
from torchvision import transforms  # type: ignore
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from torch_split.runtime import SwitchboardRuntime


resource = Resource.create({"service.name": "torchsplit-runtime"})
trace_provider = TracerProvider(resource=resource)
span_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
trace.set_tracer_provider(trace_provider)

metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4318/v1/metrics")
metric_reader = PeriodicExportingMetricReader(metric_exporter)
meter_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
metrics.set_meter_provider(meter_provider)


x = SwitchboardRuntime(Path("/dev/shm/clipfullwrapper_bs_1.tspartd"))

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
original_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
original_model = original_model.to("cuda")
original_model.eval()

# basic input
img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(1)]
texts = [f"a plain white square {9}" for i in range(1)]
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

for i in range(1):
    result, _ = x.interpret(
        debug=True,
        l_pixel_values_=pixel_values,
        l_input_ids_=input_ids,
        l_attention_mask_=attention_mask,
    )
    print(result["C"][0].shape)
original_output = original_model(
    pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
).logits_per_image
print(original_output.shape)

# print("original", original_output)
# print("split", result.keys())
# print("split output", result["C"])

# # # Calculate error metrics
# # abs_diff = torch.abs(original_output.to("cuda") - result)
# # rel_diff = abs_diff / torch.abs(original_output.to("cuda"))
# # print(f"Absolute difference: {abs_diff.item():.6f}")
# # print(f"Relative difference: {rel_diff.item():.4%}")
