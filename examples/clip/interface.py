import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor  # type: ignore
from torch_split.core import SplitClient
from examples.clip.dataset import get_dataloader


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
        self.model = CLIPFullWrapper(CLIPModel.from_pretrained("openai/clip-vit-base-patch32"))  # type: ignore
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_dataloader(self, batch_size: int) -> DataLoader:
        return get_dataloader(batch_size)
