import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from torch_split.client import InstrumentedModule, SplitClient


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

        self.model = CLIPFullWrapper(model).to("cuda:0")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_example_inputs(self) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(16)]
        texts = ["a plain white square" for _ in range(16)]

        enc = self.processor(images=img, text=texts, return_tensors="pt", padding=True)  # type: ignore
        pixel_values = enc["pixel_values"].to("cuda:0")
        input_ids = enc["input_ids"].to("cuda:0")
        attention_mask = enc["attention_mask"].to("cuda:0")

        return (pixel_values, input_ids, attention_mask), {}

    def run_benchmark(self, module: InstrumentedModule):
        """Run benchmarks with different batch sizes for CLIP model."""
        device = next(self.model.parameters()).device

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8, 16]:
            print(f"Running benchmark with batch_size={batch_size}")

            with module(batch_size) as m:
                # Generate batch inputs
                imgs = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(batch_size)]
                texts = [f"a plain white square {i}" for i in range(batch_size)]

                # Process inputs
                enc = self.processor(images=imgs, text=texts, return_tensors="pt", padding=True)  # type: ignore
                pixel_values = enc["pixel_values"].to(device)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Run multiple iterations for stable measurements
                for _ in range(128):  # 10 warmup + measurement runs
                    m.run(pixel_values, input_ids, attention_mask)
