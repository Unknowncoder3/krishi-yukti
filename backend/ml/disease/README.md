# Krishi Yukti Disease AI

Krishi Yukti uses transfer learning with MobileNetV3-Small for plant-leaf disease classification.

## Dataset

Use the PlantVillage dataset. The official dataset repository contains 54,306 healthy/diseased leaf images across 14 crop species and can be obtained from the official repository:
https://github.com/spMohanty/PlantVillage-Dataset

For the current training pipeline, place the class folders containing RGB images under:

```text
backend/ml/data/plantvillage/color/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Apple___healthy/
├── Tomato___Early_blight/
├── Tomato___Late_blight/
├── Tomato___healthy/
└── ...
```

The class-folder names become the model labels.

## Train

From `backend/` with the virtual environment activated:

```bash
python -m ml.disease.train --data-dir ml/data/plantvillage/color --epochs 10 --batch-size 32
```

On an Apple Silicon Mac the trainer automatically prefers MPS, then CUDA, then CPU.

The best checkpoint is written locally to:

```text
backend/ml/disease/artifacts/disease_model.pt
```

Model artifacts and datasets are intentionally ignored by Git.

## Predict through the API

Start FastAPI from `backend/`:

```bash
uvicorn app.main:app --reload
```

Then open `/docs` and call:

```text
POST /api/v1/ai/disease/predict
```

Upload a JPG/PNG leaf image. The response contains the predicted crop, disease/health condition, confidence, and top three predictions.

## Important limitation

PlantVillage images are largely controlled/background-limited compared with real farm photographs. This model is a research/demo decision-support component, not a replacement for an agronomist or field diagnosis. Later Krishi Yukti versions should add field images, confidence calibration, out-of-distribution detection, and crop/weather context.
