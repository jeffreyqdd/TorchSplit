import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor  # type: ignore

from torch_split.interface import SplitClient


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
        out = self.core(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        return out.logits_per_image


class ClipInterface(SplitClient):
    def __init__(self):
        super().__init__()
        self.device = self.get_best_device()
        self.model = CLIPFullWrapper(CLIPModel.from_pretrained("openai/clip-vit-base-patch32"))  # type: ignore
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()  # Ensure wrapper is in eval mode

    def get_model(self) -> torch.nn.Module:
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8, 16, 32]

    def get_benchmarks(self, batch_size: int):
        def get_example_inputs(bs: int):
            while True:
                img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(bs)]
                texts = [f"a plain white square {i}" for i in range(bs)]
                enc = self.processor(
                    images=img,
                    text=texts,
                    padding="max_length",  # type: ignore
                    max_length=32,  # type: ignore
                    return_tensors="pt",  # type: ignore
                )  # type: ignore
                pixel_values = enc["pixel_values"].to(self.device)  # type: ignore
                input_ids = enc["input_ids"].to(self.device)  # type: ignore
                attention_mask = enc["attention_mask"].to(self.device)  # type: ignore

                yield (pixel_values, input_ids, attention_mask), {}

        return 10, 30, get_example_inputs(batch_size)


# if __name__ == "__main__":
#     run_inference()
