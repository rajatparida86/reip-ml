from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_dir: str = "/app/models"
    log_level: str = "info"
    env: str = "development"

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()
