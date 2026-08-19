from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr

    groq_api_key: SecretStr

    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    chroma_persist_dir: Path = Path("./chroma_db")

    embedding_model: str = "intfloat/multilingual-e5-base"
    import_batch_size: int = 100
    import_max_sentences: int = 50000
    tatoeba_data_dir: Path = Path("./data/tatoeba")


@lru_cache
def get_settings() -> Settings:
    return Settings()