from app.models.sentence import SimilarSentence
from uuid import uuid4

from loguru import logger
from neo4j import AsyncDriver

from app.models.sentence import Sentence, SentenceCreate, JLPTLevel
from app.models.token import Token
from app.services.tokenizer import TokenizerService
from app.services.llm import LLMService
from app.services.embeddings import EmbeddingService
class SentenceService:
    def __init__(
        self,
        driver: AsyncDriver,
        tokenizer: TokenizerService,
        llm: LLMService,
        embeddings: EmbeddingService,
    ) -> None:
        self._driver = driver
        self._tokenizer = tokenizer
        self._llm = llm
        self._embeddings = embeddings

    async def create(self, payload: SentenceCreate, analyze: bool = True) -> Sentence:
        sentence_id = str(uuid4())
        tokens = self._tokenizer.tokenize(payload.text)
        embedding = self._embeddings.embed_passage(payload.text)

        level: JLPTLevel | None = None
        if analyze:
            try:
                analysis = await self._llm.analyze(payload.text)
                level = analysis.level
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        async with self._driver.session() as session:
            record = await session.execute_write(
                self._create_sentence_tx,
                sentence_id=sentence_id,
                payload=payload,
                tokens=tokens,
                level=level,
                embedding=embedding,
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
    async def _create_sentence_tx(
        tx,
        sentence_id: str,
        payload: SentenceCreate,
        tokens: list[Token],
        level: JLPTLevel | None,
        embedding: list[float],
    ):
        await tx.run(
            """
            CREATE (s:Sentence {
                id: $id,
                text: $text,
                translation: $translation,
                source: $source,
                level: $level,
                embedding: $embedding,
                created_at: datetime()
            })
            """,
            id=sentence_id,
            text=payload.text,
            translation=payload.translation,
            source=payload.source,
            level=level.value if level else None,
            embedding=embedding,
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

    async def find_similar(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5,
        level: JLPTLevel | None = None,
    ) -> list[SimilarSentence]:
        query_embedding = self._embeddings.embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes('sentence_embedding', $limit, $embedding)
        YIELD node AS s, score
        WHERE score >= $min_score
        AND ($level IS NULL OR s.level = $level)
        RETURN s.id AS id,
           s.text AS text,
           s.translation AS translation,
           s.level AS level,
           score
        ORDER BY score DESC
        """

        async with self._driver.session() as session:
            result = await session.run(
                cypher,
                embedding=query_embedding,
                limit=limit,
                min_score=min_score,
                level=level.value if level else None,
            )
            records = [r async for r in result]

        return [
            SimilarSentence(
                id=r["id"],
                text=r["text"],
                translation=r["translation"],
                level=r["level"],
                score=r["score"],
            )
            for r in records
        ]

    async def find_similar_using_word(
        self,
        query: str,
        word_lemma: str,
        limit: int = 10,
    ) -> list[SimilarSentence]:
        query_embedding = self._embeddings.embed_query(query)

        cypher = """
        CALL db.index.vector.queryNodes('sentence_embedding', 50, $embedding)
        YIELD node AS s, score
        MATCH (s)-[:CONTAINS]->(w:Word {lemma: $word})
        RETURN s.id AS id, s.text AS text, s.translation AS translation,
               s.level AS level, score
        ORDER BY score DESC
        LIMIT $limit
        """
        async with self._driver.session() as session:
            result = await session.run(
                cypher,
                embedding=query_embedding,
                word=word_lemma,
                limit=limit,
            )
            records = [r async for r in result]
            return [
            SimilarSentence(
                id=r["id"],
                text=r["text"],
                translation=r["translation"],
                level=r["level"],
                score=r["score"],
            )
            for r in records
        ]