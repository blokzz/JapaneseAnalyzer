from uuid import uuid4

from loguru import logger
from neo4j import AsyncDriver

from app.models.sentence import Sentence, SentenceCreate
from app.models.token import Token
from app.services.tokenizer import TokenizerService


class SentenceService:
    def __init__(self, driver: AsyncDriver, tokenizer: TokenizerService) -> None:
        self._driver = driver
        self._tokenizer = tokenizer

    async def create(self, payload: SentenceCreate) -> Sentence:
        sentence_id = str(uuid4())
        tokens = self._tokenizer.tokenize(payload.text)

        async with self._driver.session() as session:
            record = await session.execute_write(
                self._create_sentence_tx,
                sentence_id=sentence_id,
                payload=payload,
                tokens=tokens,
            )

        node = record["s"]
        logger.info(f"Created sentence {sentence_id} with {len(tokens)} tokens")
        return Sentence(
            id=node["id"],
            text=node["text"],
            translation=node.get("translation"),
            source=node.get("source"),
            level=node.get("level"),
            created_at=node["created_at"].to_native(),
        )

    @staticmethod
    async def _create_sentence_tx(tx, sentence_id: str, payload: SentenceCreate, tokens: list[Token]):
        await tx.run(
            """
            CREATE (s:Sentence {
                id: $id,
                text: $text,
                translation: $translation,
                source: $source,
                created_at: datetime()
            })
            """,
            id=sentence_id,
            text=payload.text,
            translation=payload.translation,
            source=payload.source,
        )

        token_data = [
            {
                "lemma": t.lemma,
                "reading": t.reading,
                "pos": t.pos.value,
                "position": i,
            }
            for i, t in enumerate(tokens)
        ]
        await tx.run(
            """
            MATCH (s:Sentence {id: $sentence_id})
            UNWIND $tokens AS tok
            MERGE (w:Word {lemma: tok.lemma})
                ON CREATE SET w.reading = tok.reading, w.pos = tok.pos, w.frequency = 1
                ON MATCH SET w.frequency = w.frequency + 1
            CREATE (s)-[:CONTAINS {position: tok.position}]->(w)
            """,
            sentence_id=sentence_id,
            tokens=token_data,
        )


        result = await tx.run("MATCH (s:Sentence {id: $id}) RETURN s", id=sentence_id)
        return await result.single()

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