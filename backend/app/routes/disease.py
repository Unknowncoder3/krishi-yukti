from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from io import BytesIO

router = APIRouter(prefix="/api/v1/ai/disease", tags=["Disease AI"])


@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        image = Image.open(BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    # Lazy import keeps the API usable even before ML dependencies/model artifacts exist.
    try:
        from ml.disease.predict import predict
        result = predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Disease model inference failed.") from exc

    result["message"] = "AI result is decision support, not a confirmed agricultural diagnosis."
    return result
