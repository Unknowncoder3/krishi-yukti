from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as torchvision_datasets, transforms

from ml.disease.model import build_model, select_device

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class HFDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["label"])


def make_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def make_hf_loaders(batch_size: int, num_workers: int):
    print("Loading PlantVillage from Hugging Face...")
    dataset = load_dataset("mohanty/PlantVillage", "color")
    train_base = dataset["train"]
    test_ds = dataset["test"]

    # Create validation data by splitting leaf IDs, preventing the same physical
    # leaf from appearing in both train and validation.
    leaf_ids = list(set(train_base["leaf_id"]))
    rng = random.Random(SEED)
    rng.shuffle(leaf_ids)
    split_at = int(len(leaf_ids) * 0.9)
    train_leaf_ids = set(leaf_ids[:split_at])
    val_leaf_ids = set(leaf_ids[split_at:])

    train_base = train_base.filter(lambda row: row["leaf_id"] in train_leaf_ids)
    val_base = dataset["train"].filter(lambda row: row["leaf_id"] in val_leaf_ids)

    train_tfms, eval_tfms = make_transforms()
    train_ds = HFDataset(train_base, train_tfms)
    val_ds = HFDataset(val_base, eval_tfms)
    test_ds = HFDataset(test_ds, eval_tfms)

    classes = dataset["train"].features["label"].names
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
        classes,
    )


def make_folder_loaders(data_dir: Path, batch_size: int, num_workers: int):
    train_tfms, eval_tfms = make_transforms()
    base = torchvision_datasets.ImageFolder(data_dir)
    classes = base.classes
    n = len(base)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(SEED)
    train_idx, val_idx, test_idx = torch.utils.data.random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )
    train_ds_full = torchvision_datasets.ImageFolder(data_dir, transform=train_tfms)
    eval_ds_full = torchvision_datasets.ImageFolder(data_dir, transform=eval_tfms)
    train_ds = torch.utils.data.Subset(train_ds_full, train_idx.indices)
    val_ds = torch.utils.data.Subset(eval_ds_full, val_idx.indices)
    test_ds = torch.utils.data.Subset(eval_ds_full, test_idx.indices)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
        classes,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Krishi Yukti plant disease classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("ml/data/plantvillage/color"))
    parser.add_argument("--source", choices=["huggingface", "folder"], default="huggingface")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything()
    if args.source == "huggingface":
        train_loader, val_loader, test_loader, classes = make_hf_loaders(
            args.batch_size, args.num_workers
        )
    else:
        if not args.data_dir.exists():
            raise SystemExit(f"Dataset directory not found: {args.data_dir}")
        train_loader, val_loader, test_loader, classes = make_folder_loaders(
            args.data_dir, args.batch_size, args.num_workers
        )

    device = select_device()
    print(f"Device: {device}")
    print(
        f"Classes: {len(classes)} | Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )

    model = build_model(len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=2
    )

    artifacts = Path("ml/disease/artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    best_path = artifacts / "disease_model.pt"
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": classes,
                    "architecture": "mobilenet_v3_small",
                    "image_size": 224,
                    "val_accuracy": best_val,
                },
                best_path,
            )
            (artifacts / "classes.json").write_text(json.dumps(classes, indent=2))
            print(f"Saved best model -> {best_path}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.3f}")
    print(f"Model artifact: {best_path}")


if __name__ == "__main__":
    main()
