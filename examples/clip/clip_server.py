import base64
import io
import os
from pathlib import Path
from typing import Dict
from requests import Request

import ray
import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
from transformers import CLIPProcessor  # type: ignore
from opentelemetry import metrics, trace

from torch_split.runtime import SwitchboardRuntime, setup_tracing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = FastAPI()
data_path = Path("/dev/shm/clipfullwrapper_bs_1.tspartd")


@serve.deployment(ray_actor_options={"num_gpus": 0.8}, max_ongoing_requests=1024)
class ComponentA:
    def __init__(self):
        setup_tracing()
        self.switchboard = SwitchboardRuntime(data_path, load_only=["A"])

    def single(self, args: dict) -> dict:
        for k in args.keys():
            t = torch.tensor(args[k], device="cuda").unsqueeze(0)
            args[k] = t

        _masked_fill, text_embeds_1 = self.switchboard.call("A", **args)
        return {"text_embeds_1": text_embeds_1.detach().cpu().numpy()}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01, max_concurrent_batches=32)
    async def __call__(self, batch: list) -> list:
        return list(map(self.single, batch))


@serve.deployment(ray_actor_options={"num_gpus": 0.1, "num_cpus": 2})
class ComponentB:
    def __init__(self):
        setup_tracing()
        self.switchboard = SwitchboardRuntime(data_path, load_only=["A"])
        self.switchboard = SwitchboardRuntime(data_path, load_only=["B"])

    def single(self, args: dict) -> dict:
        for k in args.keys():
            t = torch.tensor(args[k], device="cuda").unsqueeze(0)
            args[k] = t
            # print(t.shape)

        t = self.switchboard.call("B", **args)
        return {"t": t[0].detach().cpu().numpy()}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01, max_concurrent_batches=32)
    async def __call__(self, args: list[dict]) -> list[dict]:
        return list(map(self.single, args))


@serve.deployment(ray_actor_options={"num_gpus": 0.1, "num_cpus": 2})
class ComponentC:
    def __init__(self):
        setup_tracing()
        self.switchboard = SwitchboardRuntime(data_path)

    def single(self, args) -> dict:
        for k in args.keys():
            t = torch.tensor(args[k], device="cuda")
            args[k] = t
            # print(t.shape)
        results = self.switchboard.call("C", **args)
        # print("results:", results)
        return {"output": results[0].detach().cpu().numpy()}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01, max_concurrent_batches=32)
    async def __call__(self, args: list[tuple[dict, dict]]) -> list[dict]:
        return list(map(self.single, ({**a[0], **a[1]} for a in args)))


@serve.deployment(
    max_ongoing_requests=1024,
    ray_actor_options={"num_gpus": 0, "num_cpus": 16},
)
class ClipPreprocessor:
    def __init__(self):
        setup_tracing()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.01, max_concurrent_batches=8)
    async def __call__(self, images_and_texts: list) -> list:
        images = [it["image"] for it in images_and_texts]
        texts = [it["text"] for it in images_and_texts]

        batch = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=32,
            return_tensors="np",  # <-- key change
        )

        # return per-item dicts (same shape as you had), but as numpy arrays
        pv = batch["pixel_values"]
        ids = batch["input_ids"]
        am = batch["attention_mask"]

        return [
            {
                "l_pixel_values_": pv[i],
                "l_input_ids_": ids[i],
                "l_attention_mask_": am[i],
            }
            for i in range(len(images_and_texts))
        ]


@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 1}, num_replicas=8)
@serve.ingress(app)
class Pipeline:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        A_node: DeploymentHandle,
        B_node: DeploymentHandle,
        C_node: DeploymentHandle,
    ):
        setup_tracing()
        self.pre = preprocessor
        self.A = A_node
        self.B = B_node
        self.C = C_node

    def decode_image(self, b64_image: str) -> Image.Image:
        data = base64.b64decode(b64_image)
        return Image.open(io.BytesIO(data)).convert("RGB")

    @app.post("/")
    @torch.inference_mode()
    async def infer(self, request: Request):
        # decode
        with trace.get_tracer("clip-server").start_as_current_span("pipeline"):
            payload = await request.json()

            # preprocess
            pre_out = self.pre.remote({"image": self.decode_image(payload["image"]), "text": payload["text"]})

            # Pass the same ObjectRef to both components
            # Ray's reference counting handles concurrent access1
            A_handle = self.A.remote(pre_out)
            B_handle = self.B.remote(pre_out)

            # Wait for both
            A_out = await A_handle
            B_out = await B_handle
            C = await self.C.remote((A_out, B_out))

            return {"output": "done"}


# ray.init(_tracing_startup_hook="examples.clip.clip_tracing:setup_tracing")
app = Pipeline.bind(
    preprocessor=ClipPreprocessor.bind(), A_node=ComponentA.bind(), B_node=ComponentB.bind(), C_node=ComponentC.bind()
)  # type: ignore

# app = Pipeline.bind(
# preprocessor=ClipPreprocessor.bind(), A_node=ComponentA.bind(), B_node=ComponentB.bind(), C_node=ComponentC.bind()
# )  # type: ignore
