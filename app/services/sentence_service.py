from uuid import uuid4

from loguru import logger
from neo4j import AsyncDriver

from app.models.sentence import Sentence, SentenceCreate


class SentenceService:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def create(self, payload: SentenceCreate) -> Sentence:
        sentence_id = str(uuid4())
        query = """
            CREATE (s:Sentence {
                id: $id,
                text: $text,
                translation: $translation,
                source: $source,
                created_at: datetime()
            })
            RETURN s
        """
        async with self._driver.session() as session:
            result = await session.run(
                query,
                id=sentence_id,
                text=payload.text,
                translation=payload.translation,
                source=payload.source,
            )
            record = await result.single()

        if record is None:
            raise RuntimeError(f"Failed to create sentence {sentence_id}: no record returned")

        node = record["s"]
        logger.info(f"Created sentence {sentence_id}")
        return Sentence(
            id=node["id"],
            text=node["text"],
            translation=node.get("translation"),
            source=node.get("source"),
            level=node.get("level"),
            created_at=node["created_at"].to_native(),
        )

    async def get_by_id(self, sentence_id: str) -> Sentence | None:
        query = "MATCH (s:Sentence {id: $id}) RETURN s"
        async with self._driver.session() as session:
            result = await session.run(query, id=sentence_id)
            record = await result.single()

        if not record:
            return None

        node = record["s"]
        return Sentence(
            id=node["id"],
            text=node["text"],
            translation=node.get("translation"),
            source=node.get("source"),
            level=node.get("level"),
            created_at=node["created_at"].to_native(),
        )

    async def list_all(self, limit: int = 20, offset: int = 0) -> list[Sentence]:
        query = """
            MATCH (s:Sentence)
            RETURN s
            ORDER BY s.created_at DESC
            SKIP $offset LIMIT $limit
        """
        async with self._driver.session() as session:
            result = await session.run(query, limit=limit, offset=offset)
            records = [r async for r in result]

        return [
            Sentence(
                id=r["s"]["id"],
                text=r["s"]["text"],
                translation=r["s"].get("translation"),
                source=r["s"].get("source"),
                level=r["s"].get("level"),
                created_at=r["s"]["created_at"].to_native(),
            )
            for r in records
        ]