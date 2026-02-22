"""Pydantic-settings based configuration loading from environment variables.

Reads from a .env file at the project root and exposes a typed Settings
object used throughout the application.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables or .env file.

    Attributes:
        planning_api_base_url: Root URL for the planning data API.
        planning_api_auth_token: Bearer token for API authentication.
        learning_rate: Optimiser learning rate for the approval model.
        batch_size: Mini-batch size during training.
        epochs: Number of training epochs.
        embedding_dim: Dimensionality of text embeddings.
        text_encoder_model: HuggingFace model ID for sentence-transformers.
    """

    # API settings
    planning_api_base_url: str = 'https://ibex.seractech.co.uk'
    planning_api_auth_token: str = ""

    # Model hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    embedding_dim: int = 384

    # Rate limiting
    max_concurrent_requests: int = 10

    # Text encoder
    text_encoder_model: str = "all-MiniLM-L6-v2"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"

    # Evaluation output
    output_dir: str = "outputs"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Uses ``lru_cache`` so the .env file is read at most once per process.
    """
    return Settings()
