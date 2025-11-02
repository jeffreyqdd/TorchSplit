import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from torch_split.client import TorchSplitClient


# 2) Wrap to present a pure-tensor forward and a single-tensor output
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
        # Full CLIP forward; return a single tensor to avoid dicts
        out = self.core(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use logits_per_image (shape: [batch_img, batch_text])
        return out.logits_per_image


class TestInterface(TorchSplitClient):
    def __init__(self):
        super().__init__()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPFullWrapper(model).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_example_inputs(self) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        texts = ["a plain white square"]

        enc = self.processor(images=img, text=texts, return_tensors="pt", padding=True)  # type: ignore
        pixel_values = enc["pixel_values"]
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        return (pixel_values, input_ids, attention_mask), {}

    def target_device(self) -> torch.device:
        """Return the target device for the model."""
        return torch.device("cpu")


ti = TestInterface()
ti.model.compile()
# pp = partition.PartitionProvider(TestInterface())
# pp.visualize_dominance(Path("./.bin/clip"))
# pp.visualize_dataflow(Path("./.bin/clip"), True)
# pp.create_partition()
# out = tgraph.test()
# dot = graphviz.Digraph(name="Clip")
# tgraph.render_graph(dot)
# dot.render("ClipGraph")
