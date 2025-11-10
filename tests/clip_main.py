from collections.abc import Generator
from typing import Any, NoReturn

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from torch_split.client import SplitClient


class CLIPFullWrapper(nn.Module):
    def __init__(self, core: CLIPModel):
        super().__init__()
        self.core = core

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        out = self.core(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use logits_per_image (shape: [batch_img, batch_text])
        return out.logits_per_image


class TestInterface(SplitClient):
    def __init__(self):
        super().__init__()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        self.model = CLIPFullWrapper(model)  # .to("cuda:0")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_model(self) -> torch.nn.Module:
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8]

    def get_benchmarks(
        self, batch_size: int
    ) -> tuple[int, int, Generator[tuple[tuple[Any, ...], dict[str, Any]], Any, NoReturn]]:
        def get_example_inputs(bs: int):
            while True:
                img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(bs)]
                texts = [f"a plain white square {i}" for i in range(bs)]

                enc = self.processor(images=img, text=texts, return_tensors="pt", padding=True)  # type: ignore
                pixel_values = enc["pixel_values"]
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                yield (pixel_values, input_ids, attention_mask), {}

        return 10, 30, get_example_inputs(batch_size)
