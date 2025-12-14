import os
import sys
from pathlib import Path
from torch_split import lib
from torch_split.runtime import SwitchboardRuntime

from examples.clip.clip_interface import ClipInterface

if sys.platform == "linux":
    CACHE_PATH = Path("/dev/shm")
else:
    CACHE_PATH = Path(os.getcwd())

clip_interface = ClipInterface()
partition, roots = None, None


def get_model_partition_for_batch_size(bs: int, no_cache: bool = False) -> SwitchboardRuntime:
    global partition, roots
    model_name = clip_interface.get_model().__class__.__name__.lower()
    model_path = CACHE_PATH / f"{model_name}_bs_{bs}.tspartd"

    if model_path.exists() and not no_cache:
        return SwitchboardRuntime(Path(model_path))

    if partition is None or roots is None:
        partition, roots = lib.get_partion_and_roots(clip_interface)

    lib.batch_compiler(clip_interface, partition, roots, bs).save(model_path)
    return SwitchboardRuntime(Path(model_path))


get_model_partition_for_batch_size(1, True)
get_model_partition_for_batch_size(16, True)
get_model_partition_for_batch_size(32, True)
get_model_partition_for_batch_size(64, True)
