from torch_split.runtime import SwitchboardRuntime, trace_call, trace_remote
from starlette.requests import Request
from typing import Dict
from ray import serve
from PIL import Image
from transformers import CLIPProcessor  # type: ignore
import io
import ray
from fastapi import FastAPI, UploadFile, File, Form
from ray import serve
from ray.serve.handle import DeploymentHandle
from pathlib import Path
import torch

from torch_split.runtime import SwitchboardRuntime

app = FastAPI()
data_path = Path("/dev/shm/clipfullwrapper_bs_1.tspartd")


@serve.deployment(ray_actor_options={"num_gpus": 0.3})
class ComponentA:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(data_path, load_only=["A"])

    async def __call__(self, args: dict) -> dict:
        for k in args.keys():
            args[k] = args[k].to("cuda")
        _masked_fill, text_embeds_1 = self.switchboard.call("A", **args)
        return ray.put({"text_embeds_1": text_embeds_1})


@serve.deployment(ray_actor_options={"num_gpus": 0.3, "num_cpus": 2})
class ComponentB:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(data_path, load_only=["B"])

    async def __call__(self, args: dict) -> dict:
        for k in args:
            args[k] = args[k].to("cuda")
        t = self.switchboard.call("B", **args)
        return ray.put({"t": t[0].to("cpu")})


@serve.deployment(ray_actor_options={"num_gpus": 0.3, "num_cpus": 2})
class ComponentC:
    def __init__(self):
        self.switchboard = SwitchboardRuntime(data_path)

    async def __call__(self, a, b) -> dict:
        for k in a:
            a[k] = a[k].to("cuda")

        for k in b:
            b[k] = b[k].to("cuda")
        args = {**a, **b}
        results = self.switchboard.call("C", **args)
        return {"done": "done"}


@serve.deployment(max_ongoing_requests=1024, ray_actor_options={"num_gpus": 0, "num_cpus": 16})
class ClipPreprocessor:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01, max_concurrent_batches=32)
    async def __call__(self, images_and_texts: list) -> list:
        obj = [
            self.processor(
                images=it["image"],
                text=it["text"],
                padding="max_length",  # type: ignore
                max_length=32,  # type: ignore
                return_tensors="pt",  # type: ignore
            )
            for it in images_and_texts
        ]

        return [
            ray.put(
                {
                    "l_pixel_values_": o["pixel_values"],
                    "l_input_ids_": o["input_ids"],
                    "l_attention_mask_": o["attention_mask"],
                }
            )
            for o in obj
        ]


@serve.deployment(ray_actor_options={"num_gpus": 0, "num_cpus": 8})
@serve.ingress(app)
class Pipeline:
    def __init__(
        self,
        preprocessor: DeploymentHandle,
        A_node: DeploymentHandle,
        B_node: DeploymentHandle,
        C_node: DeploymentHandle,
    ):
        self.pre = preprocessor
        self.A = A_node
        self.B = B_node
        self.C = C_node

    @app.post("/clip")
    async def infer(self, image: UploadFile = File(...), text: str = Form(...)):
        # decode
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # preprocess
        pre_out = await self.pre.remote({"image": pil_img, "text": text})

        # # step A
        A_out = self.A.remote(pre_out)

        # step B
        B_out = self.B.remote(pre_out)

        # step B
        A = await A_out
        B = await B_out
        C = await self.C.remote(A, B)

        return {"output": "done"}  # C_out["output"]


app = Pipeline.bind(
    preprocessor=ClipPreprocessor.bind(), A_node=ComponentA.bind(), B_node=ComponentB.bind(), C_node=ComponentC.bind()
)  # type: ignore

# app = Pipeline.bind(
# preprocessor=ClipPreprocessor.bind(), A_node=ComponentA.bind(), B_node=ComponentB.bind(), C_node=ComponentC.bind()
# )  # type: ignore
