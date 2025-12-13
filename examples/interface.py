from collections.abc import Generator
from typing import Any, NoReturn

import torch
import torch.nn as nn
from torchvision import transforms  # type: ignore
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
)

from torch_split.interface import SplitClient


def with_hint(x):
    setattr(x, "_hello_world", "foo bar")
    return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # for 32×32 input
        self.fc2 = nn.Linear(128, 10)  # 10 classes
        self.fc3 = with_hint(nn.Linear(10, 10))  # 10 classes

        self.seq = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
        )

    def forward(self, x):
        # nonlocal state
        x = self.pool(torch.relu(self.conv1(x)))  # (16, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # (32, 8, 8)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class ToyExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = SimpleModel()
        self.model2 = SimpleModel()

    def forward(self, x):
        a = self.model1(x)
        b = self.model2(x)
        x = a + b
        return x


class ToyInterface(SplitClient):
    def __init__(self):
        super().__init__()
        self.device = self.get_best_device()
        self.model = ToyExample()
        self.model.to(self.device)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8, 16, 32, 64, 18, 256, 512]

    def get_example_inputs(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        example_input = torch.randn(1, 100)
        example_input.to(self.device)
        return (example_input,), {}

    def get_benchmarks(
        self, batch_size: int
    ) -> tuple[int, int, Generator[tuple[tuple[Any, ...], dict[str, Any]], Any, NoReturn]]:
        def get_example_inputs(bs: int):
            while True:
                yield (torch.randn(bs, 100).to(self.device),), {}

        return 10, 30, get_example_inputs(batch_size)


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


class ClipInterface(SplitClient):
    def __init__(self):
        super().__init__()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.device = self.get_best_device()
        self.model = CLIPFullWrapper(model)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()  # Ensure wrapper is in eval mode

    def get_model(self) -> torch.nn.Module:
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8, 16, 32]

    def get_benchmarks(
        self, batch_size: int
    ) -> tuple[int, int, Generator[tuple[tuple[Any, ...], dict[str, Any]], Any, NoReturn]]:
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
                pixel_values = enc["pixel_values"].to(self.device)
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                yield (pixel_values, input_ids, attention_mask), {}

        return 10, 30, get_example_inputs(batch_size)


class Complicated(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        A = x
        B = A + 1
        C = A + 1
        D = B + 1
        E = B + C
        F = D + 1
        G = E + 1
        H = F + G
        return H


class ComplicatedInterface(SplitClient):
    def __init__(self):
        super().__init__()
        self.device = self.get_best_device()
        self.model = Complicated()
        self.model.to(self.device)

    def get_model(self) -> torch.nn.Module:
        return self.model

    def batch_sizes(self) -> list[int]:
        return [1, 2, 4, 8, 16, 32, 64, 18, 256, 512]

    def get_example_inputs(
        self,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        example_input = torch.randn(1, 100)
        example_input.to(self.device)
        return (example_input,), {}

    def get_benchmarks(
        self, batch_size: int
    ) -> tuple[int, int, Generator[tuple[tuple[Any, ...], dict[str, Any]], Any, NoReturn]]:
        def get_example_inputs(bs: int):
            while True:
                yield (torch.randn(bs, 100).to(self.device),), {}

        return 10, 30, get_example_inputs(batch_size)


class PreFLMRImageProcessor:
    """
    Replacement for AutoImageProcessor for LinWeizheDragon/PreFLMR_ViT-L.
    Handles resize, crop, tensor conversion, and ImageNet normalization.
    """

    def __init__(self, size=224):
        self.size = size

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, images, return_tensors="pt"):
        # Single image or list of images
        if not isinstance(images, list):
            images = [images]

        processed = []
        for img in images:
            if isinstance(img, str):  # path
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):  # PIL
                img = img.convert("RGB")
            else:
                raise ValueError("Image must be a PIL Image or file path string")

            processed.append(self.transform(img))

        pixel_values = torch.stack(processed)

        # if return_tensors == "pt":
        return {"pixel_values": pixel_values}
        # else:
        #     return pixel_values
