from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.disease import router as disease_router

app = FastAPI(
    title="Krishi Yukti API",
    description="AI-powered agriculture ecosystem API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(disease_router)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "krishi-yukti-api"}


@app.get("/api/v1", tags=["System"])
def api_root() -> dict[str, str]:
    return {"message": "Krishi Yukti API v1"}
