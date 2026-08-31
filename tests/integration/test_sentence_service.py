import pytest

from app.models.sentence import SentenceCreate
from app.services.sentence_service import SentenceService


@pytest.mark.asyncio
class TestSentenceServiceIntegration:
    async def test_create_and_retrieve(
        self, test_driver, mock_tokenizer, mock_llm, mock_embeddings
    ):
        service = SentenceService(test_driver, mock_tokenizer, mock_llm, mock_embeddings)
        payload = SentenceCreate(text="私は学生です", translation="I am a student")

        created = await service.create(payload, analyze=False)

        assert created.id
        assert created.text == "私は学生です"
        assert created.translations == ["I am a student"]

        retrieved = await service.get_by_id(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id

    async def test_get_nonexistent_returns_none(
        self, test_driver, mock_tokenizer, mock_llm, mock_embeddings
    ):
        service = SentenceService(test_driver, mock_tokenizer, mock_llm, mock_embeddings)
        result = await service.get_by_id("nonexistent-id")
        assert result is None

    async def test_create_stores_word_relations(
        self, test_driver, mock_tokenizer, mock_llm, mock_embeddings
    ):
        service = SentenceService(test_driver, mock_tokenizer, mock_llm, mock_embeddings)
        created = await service.create(
            SentenceCreate(text="私は学生です"), analyze=False,
        )
        async with test_driver.session() as s:
            result = await s.run(
                "MATCH (s:Sentence {id: $id})-[:CONTAINS]->(w:Word) RETURN count(w) AS c",
                id=created.id,
            )
            record = await result.single()
            assert record["c"] == 4