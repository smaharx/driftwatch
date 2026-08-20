from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/driftwatch"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security
    SECRET_KEY: str = "changeme-in-production-use-32-char-random-string"

    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Notifications
    SLACK_WEBHOOK_URL: str = ""

    # Drift thresholds (industry standard defaults)
    PSI_THRESHOLD: float = 0.2  # > 0.2 = significant drift
    KS_ALPHA: float = 0.05  # p < 0.05 = statistically significant
    JS_THRESHOLD: float = 0.15  # > 0.15 = moderate drift

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
