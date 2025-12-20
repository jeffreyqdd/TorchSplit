import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # type: ignore
from PIL import Image
from transformers import CLIPModel, CLIPProcessor  # type: ignore

from torch_split import lib
from torch_split.interface import SplitClient
from torch_split.runtime import SwitchboardRuntime

if sys.platform == "linux":
    CACHE_PATH = Path("/dev/shm")
else:
    CACHE_PATH = Path(os.getcwd())


class Food101CLIPDataset(Dataset):
    def __init__(self, hf_dataset, label_names):
        self.ds = hf_dataset
        self.label_names = label_names

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        label = item["label"]

        text = f"a photo of {self.label_names[label].replace('_', ' ')}"
        return image, text


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


clip_interface = ClipInterface()
partition, roots = None, None


def get_model_partition_for_batch_size(bs: int) -> SwitchboardRuntime:
    global partition, roots
    model_name = clip_interface.get_model().__class__.__name__.lower()
    model_path = CACHE_PATH / f"{model_name}_bs_{bs}.tspartd"

    if model_path.exists():
        return SwitchboardRuntime(Path(model_path))

    if partition is None or roots is None:
        partition, roots = lib.get_partion_and_roots(clip_interface)

    lib.batch_compiler(clip_interface, partition, roots, bs).save(model_path)
    return SwitchboardRuntime(Path(model_path))


def collate_fn(args):
    processor = clip_interface.processor
    images, texts = zip(*args)
    enc = processor(
        images=images,
        text=texts,
        padding="max_length",  # type: ignore
        max_length=32,  # type: ignore
        return_tensors="pt",  # type: ignore
    )  # type: ignore
    return {
        "l_pixel_values_": enc["pixel_values"].to(clip_interface.get_best_device()),
        "l_input_ids_": enc["input_ids"].to(clip_interface.get_best_device()),
        "l_attention_mask_": enc["attention_mask"].to(clip_interface.get_best_device()),
    }


def run_inference():
    bs = 4
    ds = load_dataset("ethz/food101", split="validation")
    label_names = ds.features["label"].names  # type: ignore
    loader = DataLoader(
        Food101CLIPDataset(ds, label_names),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    switchboard = get_model_partition_for_batch_size(bs)
    print(switchboard.switchboard.layout)
    with torch.no_grad():
        for batch in tqdm(loader):
            switchboard.interpret(**batch)


if __name__ == "__main__":
    run_inference()
