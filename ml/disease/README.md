# Crop Disease Model

## Dataset

Use a labeled crop-leaf image dataset arranged as:

```text
ml/data/disease/
├── train/
│   ├── Crop___DiseaseA/
│   └── Crop___healthy/
├── val/
└── test/
```

Do not commit large datasets or trained weights to Git. Put them in `ml/data/` and `ml/disease/artifacts/`; these paths are ignored by `.gitignore`.

## Train

From the repository root:

```bash
cd backend
python -m pip install -r requirements.txt
cd ..
python -m ml.disease.train
```

The best checkpoint is written to `ml/disease/artifacts/disease_model.pt` and labels to `classes.json`.

## API

Start the backend:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Then upload an image to `POST /api/v1/ai/disease/predict` using a multipart field named `file`.

The current API is an AI-assisted prototype. Predictions must be verified by an agricultural professional before real-world treatment decisions.
