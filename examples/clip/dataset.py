from datasets import load_dataset  # type: ignore
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor  # type: ignore


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


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def collate_fn(args):
    images, texts = zip(*args)
    enc = processor(
        images=images,
        text=texts,
        padding="max_length",  # type: ignore
        max_length=32,  # type: ignore
        return_tensors="pt",  # type: ignore
    )  # type: ignore
    return enc


def get_dataset() -> Food101CLIPDataset:
    ds = load_dataset("ethz/food101", split="validation")
    label_names = ds.features["label"].names  # type: ignore
    return Food101CLIPDataset(ds, label_names)


def get_dataloader(batch_size: int, num_workers: int = 8, prefetch_factor: int = 2) -> DataLoader:
    ds = load_dataset("ethz/food101", split="validation")
    label_names = ds.features["label"].names  # type: ignore
    loader = DataLoader(
        Food101CLIPDataset(ds, label_names),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )
    return loader
