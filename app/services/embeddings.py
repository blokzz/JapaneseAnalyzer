from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:

    def __init__(self) -> None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self._model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Model loaded, embedding dimension: {self.dimension}")

    @property
    def dimension(self) -> int | None:
        return self._model.get_embedding_dimension()

    def embed_passage(self, text: str) -> list[float]:
        vector = self._model.encode(f"passage: {text}", normalize_embeddings=True)
        return vector.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True, batch_size=32)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(f"query: {text}", normalize_embeddings=True)
        return vector.tolist()