"""Application settings loaded from environment."""
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ibex_api_key: str = Field(..., validation_alias="IBEX_API_KEY")
    ibex_base_url: str = "https://ibex.seractech.co.uk"
    ibex_max_concurrency: int = 10

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }

settings = Settings()
