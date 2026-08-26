"""
Import Tatoeba JP-EN sentence pairs into Neo4j with embeddings.

Usage:
    python -m scripts.import_tatoeba --limit 10000
    python -m scripts.import_tatoeba --limit 50000 --analyze
"""
import argparse
import asyncio
import csv
from pathlib import Path
import hashlib
from loguru import logger
from tqdm.asyncio import tqdm

from app.config import get_settings
from app.db.neo4j import neo4j_client
from app.models.sentence import SentenceCreate
from app.services.embeddings import EmbeddingService
from app.services.tokenizer import TokenizerService


async def load_pairs(data_dir: Path, limit: int) -> list[tuple[str, str]]:
    sentences_path = data_dir / "sentences.csv"
    links_path = data_dir / "links.csv"

    logger.info(f"Reading {sentences_path}...")
    jp_sentences: dict[int, str] = {}
    en_sentences: dict[int, str] = {}

    with sentences_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 3:
                continue
            sid, lang, text = int(row[0]), row[1], row[2]
            if lang == "jpn":
                jp_sentences[sid] = text
            elif lang == "eng":
                en_sentences[sid] = text

    logger.info(f"Loaded {len(jp_sentences)} JP, {len(en_sentences)} EN sentences")
    logger.info(f"Reading {links_path}...")

    pairs: list[tuple[str, str]] = []
    with links_path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            src, tgt = int(row[0]), int(row[1])
            if src in jp_sentences and tgt in en_sentences:
                pairs.append((jp_sentences[src], en_sentences[tgt]))
                if len(pairs) >= limit:
                    break

    logger.info(f"Found {len(pairs)} JP-EN pairs")
    return pairs


async def import_batch(
    driver,
    tokenizer: TokenizerService,
    embeddings: EmbeddingService,
    batch: list[tuple[str, str]],
) -> int:
    unique: dict[str, dict] = {}
    for jp_text, en_text in batch:
        if jp_text in unique:
            unique[jp_text]["translations"].append(en_text)
        else:
            unique[jp_text] = {
                "text": jp_text,
                "translations": [en_text],
            }

    unique_texts = list(unique.keys())
    vectors = embeddings.embed_passages(unique_texts)

    records = []
    for (jp_text, data), vector in zip(unique.items(), vectors, strict=True):
        tokens = tokenizer.tokenize(jp_text)
        hash_id = hashlib.sha256(jp_text.encode()).hexdigest()[:16]
        records.append({
            "id": f"tatoeba-{hash_id}",
            "text": jp_text,
            "translations": data["translations"],
            "source": "tatoeba",
            "embedding": vector,
            "tokens": [
                {"lemma": t.lemma, "reading": t.reading, "pos": t.pos.value, "position": i}
                for i, t in enumerate(tokens)
            ],
        })

    async with driver.session() as session:
        await session.execute_write(_batch_insert_tx, records=records)

    return len(records)


async def _batch_insert_tx(tx, records: list[dict]) -> None:
    await tx.run(
        """
        UNWIND $records AS rec
        MERGE (s:Sentence {id: rec.id})
        ON CREATE SET s.text = rec.text,
                      s.translations = rec.translations,
                      s.source = rec.source,
                      s.embedding = rec.embedding,
                      s.created_at = datetime()
        ON MATCH SET s.translations = 
            [t IN s.translations WHERE NOT t IN rec.translations] + rec.translations
        """,
        records=records,
    )

    await tx.run(
        """
        UNWIND $records AS rec
        MATCH (s:Sentence {id: rec.id})
        UNWIND rec.tokens AS tok
        MERGE (w:Word {lemma: tok.lemma})
            ON CREATE SET w.reading = tok.reading, w.pos = tok.pos, w.frequency = 1
            ON MATCH SET w.frequency = w.frequency + 1
        MERGE (s)-[:CONTAINS {position: tok.position}]->(w)
        """,
        records=records,
    )

async def main(limit: int, batch_size: int) -> None:
    settings = get_settings()
    await neo4j_client.connect()

    tokenizer = TokenizerService()
    embeddings = EmbeddingService()

    try:
        pairs = await load_pairs(settings.tatoeba_data_dir, limit)

        total_imported = 0
        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]

        for batch in tqdm(batches, desc="Importing", unit="batch"):
            imported = await import_batch(neo4j_client.driver, tokenizer, embeddings, batch)
            total_imported += imported

        logger.success(f"Imported {total_imported} sentences into Neo4j")
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Tatoeba JP-EN pairs")
    parser.add_argument("--limit", type=int, default=1000, help="Max sentences to import")
    parser.add_argument("--batch-size", type=int, default=100, help="Sentences per transaction")
    args = parser.parse_args()

    asyncio.run(main(args.limit, args.batch_size))