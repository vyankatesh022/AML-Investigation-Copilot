"""Application configuration and settings."""

from pathlib import Path
from typing import Set
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for FinGuard AI."""

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_DIR / "raw" / "transactions_sample.csv"
    CUSTOMERS_DATA_PATH: Path = DATA_DIR / "processed" / "customers_sample.json"
    POLICIES_DATA_DIR: Path = DATA_DIR / "policies"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM Settings
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Compliance Rule Engine Thresholds
    HIGH_VALUE_THRESHOLD: float = 10000.0
    STRUCTURING_LOWER_BOUND: float = 9000.0
    STRUCTURING_UPPER_BOUND: float = 9999.99
    RAPID_DRAIN_RATIO: float = 0.90
    RAPID_VELOCITY_TX_LIMIT_24H: int = 3
    HIGH_RISK_JURISDICTIONS: Set[str] = {
        "PRK",  # North Korea
        "IRN",  # Iran
        "MMR",  # Myanmar
        "SYR",  # Syria
        "RUS",  # High-risk sanctioned
        "CYM",  # High secrecy offshore
        "PAN",  # High secrecy offshore
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
