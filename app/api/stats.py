from fastapi import APIRouter, Depends
from neo4j import AsyncDriver
from pydantic import BaseModel

from app.api.deps import get_neo4j_driver

router = APIRouter()


class Stats(BaseModel):
    total_sentences: int
    total_words: int
    total_relations: int
    sentences_by_level: dict[str, int]
    sentences_by_source: dict[str, int]
    top_words: list[dict]


@router.get("", response_model=Stats)
async def get_stats(driver: AsyncDriver = Depends(get_neo4j_driver)) -> Stats:
    async with driver.session() as session:
        counts = await (await session.run("""
            MATCH (s:Sentence) WITH count(s) AS sc
            MATCH (w:Word) WITH sc, count(w) AS wc
            MATCH ()-[r:CONTAINS]->() RETURN sc, wc, count(r) AS rc
        """)).single()

        by_level = {
            r["level"]: r["c"]
            async for r in await session.run("""
                MATCH (s:Sentence)
                RETURN coalesce(s.level, "unknown") AS level, count(s) AS c
                ORDER BY level
            """)
        }

        by_source = {
            r["source"]: r["c"]
            async for r in await session.run("""
                MATCH (s:Sentence)
                RETURN coalesce(s.source, "unknown") AS source, count(s) AS c
            """)
        }

        top_words = [
            {"lemma": r["lemma"], "pos": r["pos"], "frequency": r["freq"]}
            async for r in await session.run("""
                MATCH (w:Word)
                RETURN w.lemma AS lemma, w.pos AS pos, w.frequency AS freq
                ORDER BY freq DESC LIMIT 20
            """)
        ]

    return Stats(
        total_sentences=counts["sc"] if counts is not None else 0,
        total_words=counts["wc"] if counts is not None else 0,
        total_relations=counts["rc"] if counts is not None else 0,
        sentences_by_level=by_level,
        sentences_by_source=by_source,
        top_words=top_words,
    )