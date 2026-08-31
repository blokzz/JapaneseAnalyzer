from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.models.sentence import AnalysisResult, JLPTLevel
from app.models.token import PartOfSpeech, Token
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.tokenizer import TokenizerService


@pytest.fixture
def sample_tokens() -> list[Token]:
    return [
        Token(surface="私", lemma="私", reading="ワタシ", pos=PartOfSpeech.PRONOUN),
        Token(surface="は", lemma="は", reading="ハ", pos=PartOfSpeech.PARTICLE),
        Token(surface="学生", lemma="学生", reading="ガクセイ", pos=PartOfSpeech.NOUN),
        Token(surface="です", lemma="です", reading="デス", pos=PartOfSpeech.AUXILIARY),
    ]


@pytest.fixture
def mock_tokenizer(sample_tokens) -> MagicMock:
    tokenizer = MagicMock(spec=TokenizerService)
    tokenizer.tokenize.return_value = sample_tokens
    return tokenizer


@pytest.fixture
def mock_embeddings() -> MagicMock:
    embeddings = MagicMock(spec=EmbeddingService)
    embeddings.embed_passage.return_value = [0.1] * 768
    embeddings.embed_query.return_value = [0.1] * 768
    embeddings.embed_passages.return_value = [[0.1] * 768]
    embeddings.dimension = 768
    return embeddings


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock(spec=LLMService)
    llm.analyze.return_value = AnalysisResult(
        sentence="私は学生です",
        level=JLPTLevel.N5,
        grammar_points=["topic marker は", "copula です"],
        vocabulary=["私", "学生"],
        difficulty_score=0.1,
        explanation="Simple sentence.",
    )
    return llm


@pytest_asyncio.fixture
async def test_driver() -> AsyncIterator:
    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7688",
        auth=("neo4j", "test_password"),
    )
    await driver.verify_connectivity()
    async with driver.session() as s:
        await s.run("""
            CREATE CONSTRAINT sentence_id_unique IF NOT EXISTS
            FOR (s:Sentence) REQUIRE s.id IS UNIQUE
        """)
        await s.run("""
            CREATE CONSTRAINT word_lemma_unique IF NOT EXISTS
            FOR (w:Word) REQUIRE w.lemma IS UNIQUE
        """)

    yield driver

    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
    await driver.close()

@pytest_asyncio.fixture
async def test_app(test_driver, mock_tokenizer, mock_llm, mock_embeddings):
    from app.main import app
    from app.api.deps import (
        get_neo4j_driver, get_tokenizer, get_llm_service, get_embedding_service,
    )

    app.dependency_overrides[get_neo4j_driver] = lambda: test_driver
    app.dependency_overrides[get_tokenizer] = lambda: mock_tokenizer
    app.dependency_overrides[get_llm_service] = lambda: mock_llm
    app.dependency_overrides[get_embedding_service] = lambda: mock_embeddings

    yield app

    app.dependency_overrides.clear()


    