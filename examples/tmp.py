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

from torch_split.lib.client import SplitClient
from torch_split.lib.switchboard import Switchboard

x = Switchboard.load(Path("/dev/shm/switchboard.tspartd"))
for name, mod in x.components.items():
    mod.eval()
    print(f"--- Component {name} code ---")
    print(mod.code)
    print("---------------------------")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
original_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
original_model.eval()

# # basic input
img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(1)]
texts = [f"a plain white square {9}" for i in range(1)]
enc = processor(
    images=img,
    text=texts,
    padding="max_length",  # type: ignore
    max_length=32,  # type: ignore
    return_tensors="pt",  # type: ignore
)  # type: ignore
pixel_values = enc["pixel_values"]
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]


# # Move inputs to CUDA to match the model device
pixel_values_cuda = pixel_values.to("cuda")
input_ids_cuda = input_ids.to("cuda")
attention_mask_cuda = attention_mask.to("cuda")

result = x.interpret(
    l_pixel_values_=pixel_values_cuda,
    l_input_ids_=input_ids_cuda,
    l_attention_mask_=attention_mask_cuda,
)
original_output = original_model(
    pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
).logits_per_image

print("original", original_output)
print("split", result.keys())
print("split output", result["C"])

# # Calculate error metrics
# abs_diff = torch.abs(original_output.to("cuda") - result)
# rel_diff = abs_diff / torch.abs(original_output.to("cuda"))
# print(f"Absolute difference: {abs_diff.item():.6f}")
# print(f"Relative difference: {rel_diff.item():.4%}")
