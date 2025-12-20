from PIL import Image
import json
import time
import gc
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path

from torch.profiler import profile, ProfilerActivity

from torch_split.core import ir, utils
from torch_split.core.partition import provider
import torch_split.lib as lib
from torch_split.interface import SplitClient
from torch_split.runtime import SwitchboardRuntime


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
                pixel_values = enc["pixel_values"].to(self.device)
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                yield (pixel_values, input_ids, attention_mask), {}

        return 10, 30, get_example_inputs(batch_size)


def get_cut(split_interface: SplitClient):
    _a, _b, generator = split_interface.get_benchmarks(32)
    args, kwargs = next(generator)
    model = split_interface.get_model()
    model_name = model.__class__.__name__
    gm = utils.capture_graph(model)(*args, **kwargs)
    tg = ir.TorchGraph.from_fx_graph(gm, label=model_name)
    pp = provider.PartitionProvider(tg)
    ap = sorted(pp.all_partitions(), key=lambda p: -sum(len(sg.enclosed_region) for sg in p.subgraphs))
    # sb = pp.create_switchboard([ap[0]])
    return [ap[0]], pp


cut = None
root = None
interface = ClipInterface()

# torch.cuda.set_per_process_memory_fraction(0.5)
# maps component name to a list of (batch_size, qps, memory_usage, gpu_utilization)
benchmark_results: dict[str, tuple[int, float, float, float]] = {}
total_memory = 24

for memory_limit in [6, 12, 24]:
    percentage = memory_limit / total_memory
    torch.cuda.set_per_process_memory_fraction(percentage)
    print(f"Set memory limit to {memory_limit} GB ({percentage * 100:.1f}%)")

    for bs in [32, 64, 128]:
        if not Path(f"/dev/shm/switchboard_bs_{bs}.tspartd").exists():
            if not cut or not root:
                cut, root = get_cut(interface)
            lib.batch_compiler(interface, cut, root, batch_size=bs).save(Path(f"/dev/shm/switchboard_bs_{bs}.tspartd"))

        print("benchmkaring bs ", bs)
        _a, _b, generator = interface.get_benchmarks(bs)
        runtime = SwitchboardRuntime(Path(f"/dev/shm/switchboard_bs_{bs}.tspartd"))

        for component_name in runtime.switchboard.components.keys():
            print("benchmarking component ", component_name)

            try:
                time_delta = 0
                max_gpu_util: float = 0
                max_mem_used: float = 0
                for i in range(30):
                    args, kwargs = next(generator)
                    result, intermediates = runtime.interpret(
                        l_pixel_values_=args[0],
                        l_input_ids_=args[1],
                        l_attention_mask_=args[2],
                    )

                    inputs = intermediates[component_name]

                    torch.cuda.reset_peak_memory_stats()
                    mem_before = torch.cuda.memory_allocated()

                    time_start = time.perf_counter_ns()
                    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
                        runtime.call(component_name, **inputs)
                    torch.cuda.synchronize()  # Ensure GPU work is done
                    time_end = time.perf_counter_ns()

                    mem_after = torch.cuda.memory_allocated()
                    peak_mem = torch.cuda.max_memory_allocated()

                    # Calculate GPU utilization from profiler
                    # self_device_time_total is in microseconds
                    total_gpu_time_us = sum(evt.self_device_time_total for evt in prof.key_averages())
                    wall_clock_us = (time_end - time_start) / 1e3  # ns to us
                    gpu_util = (total_gpu_time_us / wall_clock_us) * 100 if wall_clock_us > 0 else 0

                    time_delta += time_end - time_start
                    gc.collect()
                    time.sleep(0.1)  # Give GPU time to settle between measurements
                batch_time_s = time_delta / 30 / 1e9
                print(" average latency (s): ", batch_time_s)
                print(" throughput (qps): ", bs / batch_time_s)

                # if component_name not in ideal_batch_size_and_qps or ideal_batch_size_and_qps[component_name][1] < (
                #     bs / batch_time_s
                # ):
                #     ideal_batch_size_and_qps[component_name] = (bs, bs / batch_time_s)
            except torch.cuda.OutOfMemoryError as e:
                print("OOM at batch size ", bs, " component ", component_name)
                continue

            print("waiting")
            torch.cuda.empty_cache()
            gc.collect()
            gc.collect()
            gc.collect()
            time.sleep(2)
            gc.collect()
            gc.collect()
            gc.collect()
# print(json.dumps(ideal_batch_size_and_qps, indent=2))

# bs_16 = lib.batch_compiler(interface, cut, batch_size=1)
# bs_32 = lib.batch_compiler(interface, cut, batch_size=1)
# bs_64 = lib.batch_compiler(interface, cut, batch_size=1)
# bs_128 = lib.batch_compiler(interface, cut, batch_size=1)

# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# # original_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# # original_model.eval()

# # # # basic input
# img = [Image.new("RGB", (224, 224), color=(255, 255, 255)) for _ in range(128)]
# texts = [f"a plain white square {9}" for i in range(128)]
# enc = processor(
#     images=img,
#     text=texts,
#     padding="max_length",  # type: ignore
#     max_length=32,  # type: ignore
#     return_tensors="pt",  # type: ignore
# )  # type: ignore
# pixel_values = enc["pixel_values"].to("cuda")
# input_ids = enc["input_ids"].to("cuda")
# attention_mask = enc["attention_mask"].to("cuda")

# for i in range(1000):
#     result = x.interpret(
#         l_pixel_values_=pixel_values,
#         l_input_ids_=input_ids,
#         l_attention_mask_=attention_mask,
#     )
