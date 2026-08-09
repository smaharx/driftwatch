# core/config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/driftwatch"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "changeme_in_production"
    ENVIRONMENT: str = "development"
    SLACK_WEBHOOK_URL: str = ""

    # Drift detection thresholds
    PSI_THRESHOLD: float = 0.2
    KS_ALPHA: float = 0.05
    JS_THRESHOLD: float = 0.15

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
