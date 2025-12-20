import base64
import csv
import io
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import ray
import torch
from fastapi import FastAPI
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
from transformers import CLIPModel, CLIPProcessor

app = FastAPI()


@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 1, "num_cpus": 2},
    max_ongoing_requests=256,
)
class ClipMonolithic:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.model.eval()

    @torch.inference_mode()
    def single(self, enc: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        for key, value in enc.items():
            enc[key] = torch.tensor(value, device="cuda").unsqueeze(0)

        out = self.model(**enc)

        # to emulate some processing (same as optimized version)
        img = out.image_embeds.detach().cpu()
        txt = out.text_embeds.detach().cpu()

        return {"output": "done"}

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=0.1)
    async def __call__(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return list(map(self.single, payloads))


@serve.deployment(
    num_replicas=8,
    max_ongoing_requests=256,
    ray_actor_options={"num_gpus": 0, "num_cpus": 8},
)
class ClipPreprocessor:
    def __init__(self):
        # setup_tracing()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("ClipPreprocessor loaded")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def __call__(self, images_and_texts: list) -> list:
        images = [it["image"] for it in images_and_texts]
        texts = [it["text"] for it in images_and_texts]

        # Process entire batch at once
        batch = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=32,
            return_tensors="np",
        )

        # Return per-item dicts as numpy arrays
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        return [
            {
                "pixel_values": pixel_values[i],
                "input_ids": input_ids[i],
                "attention_mask": attention_mask[i],
            }
            for i in range(len(images_and_texts))
        ]


@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 0.1}, num_replicas=64)
@serve.ingress(app)
class Pipeline:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        monolithic: DeploymentHandle,
    ):
        # setup_tracing()
        self.pre = preprocessor
        self.monolithic = monolithic

    def decode_image(self, b64_image: str) -> Image.Image:
        data = base64.b64decode(b64_image)
        return Image.open(io.BytesIO(data)).convert("RGB")

    @app.post("/clip")
    @torch.inference_mode()
    async def infer(self, request: Request):
        payload = await request.json()

        # preprocess
        pre_out = await self.pre.remote({"image": self.decode_image(payload["image"]), "text": payload["text"]})
        output = await self.monolithic.remote(pre_out)

        return {"output": "done"}


app = Pipeline.bind(ClipPreprocessor.bind(), ClipMonolithic.bind())
