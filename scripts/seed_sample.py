"""Load sample sentences for quick demo (no Tatoeba required)."""
import asyncio
import json
from pathlib import Path

from loguru import logger

from app.db.neo4j import neo4j_client
from app.services.embeddings import EmbeddingService
from app.services.tokenizer import TokenizerService
from scripts.import_tatoeba import import_batch


SAMPLE_PATH = Path("data/sample_sentences.jsonl")


async def main() -> None:
    await neo4j_client.connect()
    tokenizer = TokenizerService()
    embeddings = EmbeddingService()

    try:
        samples = [
            json.loads(line)
            for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs = [(s["text"], s["translation"]) for s in samples]

        logger.info(f"Loading {len(pairs)} sample sentences...")
        imported = await import_batch(
            neo4j_client.driver, tokenizer, embeddings, pairs
        )
        logger.success(f"Loaded {imported} sample sentences")
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(main())