from fastapi import FastAPI

app = FastAPI(
    title="Krishi Yukti API",
    description="AI-powered agriculture ecosystem API",
    version="0.1.0",
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "krishi-yukti-api"}


@app.get("/api/v1", tags=["System"])
def api_root() -> dict[str, str]:
    return {"message": "Krishi Yukti API v1"}
