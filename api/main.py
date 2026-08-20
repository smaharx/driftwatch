# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import models
from core.config import settings

app = FastAPI(
    title="DriftWatch API",
    description="ML data quality and drift monitoring platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(models.router)


@app.get("/health", tags=["system"])
def health_check() -> dict:
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/", tags=["system"])
def root() -> dict:
    return {"message": "DriftWatch API", "docs": "/docs"}
