from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ml.disease.model import build_model, select_device


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "disease_model.pt"

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _parse_label(label: str) -> tuple[str, str]:
    if "___" in label:
        crop, condition = label.split("___", 1)
    else:
        crop, condition = "Unknown", label
    condition = condition.replace("_", " ").strip()
    if condition.lower() == "healthy":
        condition = "Healthy"
    return crop.replace("_", " "), condition


@lru_cache(maxsize=1)
def _load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Disease model is not trained yet. Run: "
            "python -m ml.disease.train"
        )

    device = select_device()
    checkpoint = torch.load(ARTIFACT_PATH, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, device


def predict(image: Image.Image) -> dict:
    model, class_names, device = _load_model()
    image = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))

    predictions = []
    for probability, index in zip(top_probs.tolist(), top_indices.tolist()):
        crop, condition = _parse_label(class_names[index])
        predictions.append({
            "crop": crop,
            "disease": condition,
            "confidence": round(probability, 4),
        })

    best = predictions[0]
    return {
        "crop": best["crop"],
        "disease": best["disease"],
        "confidence": best["confidence"],
        "top_predictions": predictions,
    }
