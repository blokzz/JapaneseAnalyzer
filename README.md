# 🇯🇵 Japanese RAG

GraphRAG backend for learning Japanese — combines semantic search, LLM analysis, 
and a knowledge graph of words, sentences, and kanji into a single queryable API.

<p align="center">
  <img src="docs/images/demo.gif" width="700" alt="Demo">
</p>

## What it does

- 🔍 **Cross-lingual semantic search** — query in English, Polish, or any language;
  find Japanese sentences by meaning
- 🧠 **Automatic JLPT assessment** — sentences graded N5→N1 by LLM with structured output
- 📊 **Knowledge graph** — sentences, dictionary-form words, and kanji linked with 
  frequency counts and grammatical roles
- 🤖 **LLM-generated flashcards** — POST any Japanese text, get structured cards 
  (vocabulary / phrase / grammar / kanji / onomatopoeia) with examples
- 🎴 **Flashcard app integration** — approved cards POSTed to `/ingest` are ingested 
  into the graph; ref id returned for the client to store
- 🎌 **Real dataset** — ships importer for 26k+ sentences from Tatoeba

## Stack

- **Neo4j 5** — graph database with built-in vector search
- **FastAPI** — async HTTP API with automatic OpenAPI docs
- **Groq (Llama 3.3 70B)** — free-tier LLM with JSON mode
- **sentence-transformers (multilingual E5)** — 768-dim embeddings
- **Fugashi / MeCab (unidic-lite)** — Japanese tokenization with lemmatization
- **Pydantic v2** — validation on every boundary
- **Docker Compose** — one command runs Neo4j + API together

## Data model

    (Sentence {text, translations[], level, embedding})
         │ :CONTAINS {position}
         ▼
    (Word {lemma, reading, pos, type, frequency})
         │ :CONTAINS_KANJI
         ▼
    (Kanji {character})

Neo4j holds both the graph AND vector embeddings — one database, unified queries.
You can filter semantic search results by graph structure in a single Cypher query
(e.g. *"N3 sentences semantically about food that contain the verb 食べる"*).

Card ingest is idempotent — the same input always produces the same node ids
and merges into existing nodes.

## Quick start

Requires Docker Desktop and a free [Groq API key](https://console.groq.com).

​```bash
git clone https://github.com/<you>/japanese-rag.git
cd japanese-rag
cp .env.example .env      # add your GROQ_API_KEY
docker compose up -d
docker compose exec api python -m scripts.seed_sample
​```

Then open:
- API docs: http://localhost:8000/docs
- Neo4j browser: http://localhost:7474 (user: `neo4j`)

## Try it

​```bash
# Cross-lingual semantic search
curl "http://localhost:8000/sentences/similar?q=I%20love%20ramen"
curl "http://localhost:8000/sentences/similar?q=food&level=N3&limit=5"

# Analyze a sentence — JLPT level, grammar points, vocabulary
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "私は寿司が食べたい"}'

# Generate flashcards from any text
curl -X POST http://localhost:8000/flashcards \
  -H "Content-Type: application/json" \
  -d '{"text": "食べる"}'

# Ingest an approved card into the knowledge graph
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "main_text": "食べる",
    "card_type": "vocabulary",
    "translation": "to eat",
    "jlpt": "N5",
    "examples": [{"sentence": "寿司を食べる", "translation": "eat sushi"}]
  }'
​```

## Full dataset (Tatoeba)

`scripts/seed_sample` loads a handful for smoke testing. For the real dataset:

​```bash
# Download from https://tatoeba.org/eng/downloads (sentences + links)
# Extract to data/tatoeba/
docker compose exec api python -m scripts.import_tatoeba --limit 30000
​```

~87% unique Japanese sentences (multiple translations merge into a single node).
Data © Tatoeba contributors under [CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).

## Development

Local dev without Docker for the API — just Neo4j in a container:

​```bash
docker compose up -d neo4j

# Windows PowerShell
python -m venv venv; .\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv && source venv/bin/activate

pip install -e ".[dev]"
uvicorn app.main:app --reload

# Tests
pytest tests/unit -v                                        # fast, mocked
docker compose --profile test up -d neo4j-test
pytest tests/integration -v
​```

## Related projects

[NihongoCards](https://github.com/<you>/nihongo-cards) — Tauri + React desktop 
flashcard app with FSRS spaced repetition. Cards approved there are POSTed to 
`/ingest` and their returned `word_id` is stored as a pointer to the knowledge graph.

## License

MIT