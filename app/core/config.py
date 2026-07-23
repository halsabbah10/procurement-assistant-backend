from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "procurement"
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    allowed_origins: str = "http://localhost:5173"
    environment: str = "development"
    rate_limit_per_minute: int = 20
    daily_request_cap: int = 200

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
