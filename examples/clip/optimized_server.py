import base64
import io
import os
from pathlib import Path
from typing import Any, Dict

import ray
import torch
from fastapi import FastAPI, File, Form, UploadFile
from opentelemetry import metrics, trace
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle
from requests import Request
from starlette.requests import Request
from transformers import CLIPProcessor  # type: ignore

from torch_split.runtime import SwitchboardRuntime

os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = FastAPI()
data_path = Path("/dev/shm/switchboard.tspartd")


def _apply_recursive(obj: Any, fn: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _apply_recursive(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        res = [_apply_recursive(v, fn) for v in obj]
        return tuple(res) if isinstance(obj, tuple) else res
    return fn(obj)


def _move_to_device(obj: Any, device: torch.device, **kwargs) -> Any:
    return _apply_recursive(obj, lambda x: x.to(device, **kwargs) if hasattr(x, "to") else x)


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.25, "num_cpus": 4}, max_ongoing_requests=1024)
class ComponentA:
    def __init__(self):
        self.switchboard = SwitchboardRuntime.from_path(data_path, load_only=["A"])
        self.switchboard.switchboard.to_device(torch.device("cuda"))
        print("Component A loaded")

    def single(self, args: dict) -> dict:
        for key, value in args.items():
            args[key] = torch.tensor(value, device="cuda").unsqueeze(0)
        text_embeds_1 = self.switchboard.call("A", **args)["text_embeds_1"]
        return {"text_embeds_1": text_embeds_1.detach().cpu().numpy()}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def __call__(self, batch: list) -> list:
        return list(map(self.single, batch))


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.25, "num_cpus": 4}, max_ongoing_requests=1024)
class ComponentB:
    def __init__(self):
        # setup_tracing()
        self.switchboard = SwitchboardRuntime.from_path(data_path, load_only=["B"])
        self.switchboard.switchboard.to_device(torch.device("cuda"))
        print("Component B loaded")

    def single(self, args: dict) -> dict:
        for key, value in args.items():
            args[key] = torch.tensor(value, device="cuda").unsqueeze(0)
        t = self.switchboard.call("B", **args)["t"]
        # print(t)
        return {"t": t.detach().cpu().numpy()}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def __call__(self, args: list[dict]) -> list[dict]:
        return list(map(self.single, args))


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.2, "num_cpus": 4}, max_ongoing_requests=1024)
class ComponentC:
    def __init__(self):
        # setup_tracing()
        self.switchboard = SwitchboardRuntime.from_path(data_path, load_only=["C"])
        self.switchboard.switchboard.to_device(torch.device("cuda"))
        print("Component C loaded")

    def single(self, args) -> dict:
        for key, value in args.items():
            args[key] = torch.tensor(value, device="cuda")
            # print(args[key].shape)
        output = self.switchboard.call("C", **args)["output"]
        # print(output)
        return {"output": output[0].detach().cpu().numpy()}

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def __call__(self, args: list[tuple[dict, dict]]) -> list[dict]:
        return list(map(self.single, ({**a[0], **a[1]} for a in args)))


@serve.deployment(
    num_replicas=2,
    max_ongoing_requests=1,
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

        batch = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=32,
            return_tensors="np",
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


@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 8}, num_replicas=1)
@serve.ingress(app)
class Pipeline:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        A_node: DeploymentHandle,
        B_node: DeploymentHandle,
        C_node: DeploymentHandle,
    ):
        # setup_tracing()
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
