"""Train a crop disease classifier.

Expected dataset layout:
ml/data/disease/train/<class_name>/*.jpg
ml/data/disease/val/<class_name>/*.jpg
ml/data/disease/test/<class_name>/*.jpg

This starter uses transfer learning with MobileNetV3 and saves a PyTorch checkpoint.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1] / "data" / "disease"
OUT = Path(__file__).resolve().parent / "artifacts"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    if not (ROOT / "train").exists():
        raise FileNotFoundError(f"Dataset not found: {ROOT}. Add train/val/test folders first.")

    train_ds = datasets.ImageFolder(ROOT / "train", transform=transform)
    val_ds = datasets.ImageFolder(ROOT / "val", transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(train_ds.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"epoch={epoch + 1}/{EPOCHS} val_accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "num_classes": len(train_ds.classes)}, OUT / "disease_model.pt")
            (OUT / "classes.json").write_text(json.dumps(train_ds.classes, indent=2))

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
