from neo4j import AsyncDriver

from app.db.neo4j import neo4j_client
from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService
from app.services.tokenizer import TokenizerService

_embeddings_singleton: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = EmbeddingService()
    return _embeddings_singleton
    
_llm_singleton: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton

def get_neo4j_driver() -> AsyncDriver:
    return neo4j_client.driver


_tokenizer_singleton: TokenizerService | None = None


def get_tokenizer() -> TokenizerService:
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        _tokenizer_singleton = TokenizerService()
    return _tokenizer_singleton
