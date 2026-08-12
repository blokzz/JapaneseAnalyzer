from neo4j import AsyncDriver

from app.db.neo4j import neo4j_client
from app.services.llm import LLMService

_llm_singleton: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton

def get_neo4j_driver() -> AsyncDriver:
    return neo4j_client.driver