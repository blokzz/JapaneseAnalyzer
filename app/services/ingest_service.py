import hashlib

from loguru import logger
from neo4j import AsyncDriver

from app.models.ingest import CardIngestPayload, IngestResult
from app.models.token import PartOfSpeech
from app.services.embeddings import EmbeddingService
from app.services.tokenizer import TokenizerService
from app.services.kanji import extract_kanji


CONTENT_POS = {PartOfSpeech.NOUN, PartOfSpeech.VERB, PartOfSpeech.ADJECTIVE}


class IngestService:
    def __init__(self, driver, tokenizer, embeddings):
        self._driver = driver
        self._tokenizer = tokenizer
        self._embeddings = embeddings

    async def ingest(self, payload: CardIngestPayload) -> IngestResult:
        main_tokens = self._tokenizer.tokenize(payload.main_text)
        if len(main_tokens) == 1:
            word_id = main_tokens[0].lemma
        else:
            content = [t for t in main_tokens if t.pos in {PartOfSpeech.NOUN, PartOfSpeech.VERB, PartOfSpeech.ADJECTIVE}]
            word_id = content[0].lemma if content else main_tokens[0].lemma

        example_data = [
            self._prepare_example(ex) for ex in payload.examples
        ]

        all_kanji = list({
            ch
            for text in [payload.main_text, *[ex.sentence for ex in payload.examples]]
            for ch in extract_kanji(text)
        })

        async with self._driver.session() as session:
            await session.execute_write(
                self._ingest_tx,
                payload=payload,
                main_tokens=main_tokens,
                example_data=example_data,
                all_kanji=all_kanji,
            )

        logger.info(f"Ingested card word_id={word_id} examples={len(example_data)}")
        return IngestResult(word_id=word_id)

    # @staticmethod
    # def _pick_primary_lemma(tokens, fallback: str) -> str:
    #     if not tokens:
    #         return fallback
    #     content = [t for t in tokens if t.pos in CONTENT_POS]
    #     return (content[0] if content else tokens[0]).lemma

    def _prepare_example(self, ex) -> dict:
        tokens = self._tokenizer.tokenize(ex.sentence)
        return {
            "id": f"card-{hashlib.sha256(ex.sentence.encode()).hexdigest()[:16]}",
            "text": ex.sentence,
            "translation": ex.translation,
            "embedding": self._embeddings.embed_passage(ex.sentence),
            "tokens": [
                {"lemma": t.lemma, "reading": t.reading, "pos": t.pos.value, "position": i}
                for i, t in enumerate(tokens)
            ],
            "word_kanji": [
                {"lemma": t.lemma, "chars": extract_kanji(t.lemma)}
                for t in tokens if extract_kanji(t.lemma)
            ],
        }

    @staticmethod
    async def _ingest_tx(tx, payload, main_tokens, example_data, all_kanji):
        main_token_data = [
            {"lemma": t.lemma, "reading": t.reading, "pos": t.pos.value}
            for t in main_tokens
        ]
        await tx.run("""
            UNWIND $tokens AS tok
            MERGE (w:Word {lemma: tok.lemma})
            ON CREATE SET w.reading = tok.reading, w.pos = tok.pos,
                          w.frequency = 1, w.created_at = datetime()
            ON MATCH SET w.frequency = w.frequency + 1
        """, tokens=main_token_data)

        if example_data:
            await tx.run("""
                UNWIND $examples AS ex
                MERGE (s:Sentence {id: ex.id})
                ON CREATE SET s.text = ex.text,
                              s.translations = CASE WHEN ex.translation IS NULL
                                                     THEN [] ELSE [ex.translation] END,
                              s.source = "newflashcards",
                              s.embedding = ex.embedding,
                              s.level = $jlpt,
                              s.created_at = datetime()
                ON MATCH SET s.translations = CASE
                    WHEN ex.translation IS NULL THEN s.translations
                    WHEN ex.translation IN s.translations THEN s.translations
                    ELSE s.translations + ex.translation END

                WITH s, ex
                UNWIND ex.tokens AS tok
                MERGE (w:Word {lemma: tok.lemma})
                ON CREATE SET w.reading = tok.reading, w.pos = tok.pos,
                              w.frequency = 1, w.created_at = datetime()
                ON MATCH SET w.frequency = w.frequency + 1
                MERGE (s)-[:CONTAINS {position: tok.position}]->(w)
            """, examples=example_data,
                 jlpt=payload.jlpt.value if payload.jlpt else None)

        if all_kanji:
            await tx.run("""
                UNWIND $chars AS ch
                MERGE (k:Kanji {character: ch})
                ON CREATE SET k.created_at = datetime()
            """, chars=all_kanji)

        all_word_kanji = [
            {"lemma": t.lemma, "chars": extract_kanji(t.lemma)}
            for t in main_tokens if extract_kanji(t.lemma)
        ] + [item for ex in example_data for item in ex["word_kanji"]]
        
        seen = set()
        unique_wk = [x for x in all_word_kanji if not (x["lemma"] in seen or seen.add(x["lemma"]))]
        
        if unique_wk:
            await tx.run("""
                UNWIND $items AS item
                MATCH (w:Word {lemma: item.lemma})
                UNWIND item.chars AS ch
                MATCH (k:Kanji {character: ch})
                MERGE (w)-[:CONTAINS_KANJI]->(k)
            """, items=unique_wk)