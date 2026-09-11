"""Inference helper for the trained disease model."""
from pathlib import Path
import json
import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    checkpoint_path = ARTIFACTS / "disease_model.pt"
    classes_path = ARTIFACTS / "classes.json"
    if not checkpoint_path.exists() or not classes_path.exists():
        raise FileNotFoundError("Train the disease model first; model artifacts are missing.")
    classes = json.loads(classes_path.read_text())
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


def predict(image: Image.Image):
    model, classes = load_model()
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]
    index = int(probabilities.argmax())
    return {"disease": classes[index], "confidence": round(float(probabilities[index]), 4)}
