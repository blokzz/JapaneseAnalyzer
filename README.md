# 🇯🇵 Japanese RAG

GraphRAG backend for learning Japanese — combines semantic search, LLM analysis,
and a knowledge graph of words, sentences, and kanji into a single queryable API.

<!--
<p align="center">
  <img src="docs/images/demo.gif" width="700" alt="Demo">
</p>
-->

## What it does

- 🔍 **Cross-lingual semantic search** — query in English, Polish, or any language;
  find Japanese sentences by meaning
- 🧠 **Automatic JLPT assessment** — sentences graded N5 → N1 by LLM with structured output
- 📊 **Knowledge graph** — sentences, dictionary-form words, and kanji linked with
  frequency counts and grammatical roles
- 🤖 **LLM-generated flashcards** — POST any Japanese text, get structured cards
  (vocabulary / phrase / grammar / kanji / onomatopoeia) with examples
- 🎴 **Flashcard app integration** — approved cards POSTed to `/ingest` are merged
  into the graph; `word_id` returned for the client to store
- ✂️ **Japanese tokenization** — morphological analysis via Fugashi/MeCab with
  lemmatization, readings, and POS tagging
- 📈 **Graph statistics** — live counts of sentences, words, relations, JLPT
  distribution, and top-frequency words
- 🎌 **Real dataset** — ships importer for 26 k+ sentences from Tatoeba

## Stack

| Layer | Technology |
|---|---|
| Graph DB | **Neo4j 5** — nodes, relationships, and vector index in one store |
| Vector search | Neo4j built-in vector index (768-dim, cosine) |
| API | **FastAPI** — async HTTP with automatic OpenAPI docs |
| LLM | **Groq** (Llama 3.3 70B) — free-tier, JSON mode |
| Embeddings | **sentence-transformers** (`intfloat/multilingual-e5-base`) |
| Tokenizer | **Fugashi / MeCab** (`unidic-lite`) — lemmatization + POS |
| Validation | **Pydantic v2** — strict models on every boundary |
| Infra | **Docker Compose** — one command runs Neo4j + API |

## Data model

```
(Sentence {text, translations[], level, embedding})
     │ :CONTAINS {position}
     ▼
(Word {lemma, reading, pos, type, frequency})
     │ :CONTAINS_KANJI
     ▼
(Kanji {character})
```

Neo4j holds both the graph **and** vector embeddings — one database, unified queries.
You can filter semantic search results by graph structure in a single Cypher query
(e.g. *"N3 sentences semantically about food that contain the verb 食べる"*).

Card ingest is idempotent — the same input always produces the same node ids
and merges into existing nodes.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/sentences/similar?q=…` | Cross-lingual semantic search (optional `level`, `min_score`, `limit`) |
| `GET` | `/sentences/similar-with-word?q=…&word=…` | Semantic search filtered by a specific word |
| `GET` | `/sentences/{id}` | Get sentence by id |
| `GET` | `/sentences?limit=&offset=` | List sentences (paginated) |
| `POST` | `/sentences` | Create a sentence (with optional LLM analysis) |
| `POST` | `/analyze` | JLPT analysis — level, grammar points, vocabulary, difficulty |
| `POST` | `/analyze/cards` | Generate flashcards from any Japanese text |
| `POST` | `/ingest` | Ingest an approved card into the knowledge graph |
| `POST` | `/tokenize` | Morphological tokenization (surface, lemma, reading, POS) |
| `GET` | `/stats` | Graph statistics — counts, JLPT distribution, top words |

Full interactive docs available at `http://localhost:8000/docs` after starting the API.

## Quick start

**Requirements:** Docker Desktop and a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/blokzz/japanese-rag.git
cd japanese-rag
cp .env.example .env      # add your GROQ_API_KEY
docker compose up -d
docker compose exec api python -m scripts.seed_sample
```

Then open:

| Service | URL |
|---|---|
| API docs (Swagger) | http://localhost:8000/docs |
| Neo4j browser | http://localhost:7474 (user: `neo4j`) |

## Try it

```bash
# Cross-lingual semantic search
curl "http://localhost:8000/sentences/similar?q=I%20love%20ramen"
curl "http://localhost:8000/sentences/similar?q=food&level=N3&limit=5"

# Analyze a sentence — JLPT level, grammar points, vocabulary
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "私は寿司が食べたい"}'

# Generate flashcards from any text
curl -X POST http://localhost:8000/analyze/cards \
  -H "Content-Type: application/json" \
  -d '{"text": "食べる"}'

# Tokenize Japanese text (morphological analysis)
curl -X POST http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "私は寿司が食べたい"}'

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

# Graph statistics
curl http://localhost:8000/stats
```

## Full dataset (Tatoeba)

`scripts/seed_sample` loads a handful of sentences for smoke testing.
For the real dataset:

```bash
# Download from https://tatoeba.org/eng/downloads (sentences + links)
# Extract to data/tatoeba/
docker compose exec api python -m scripts.import_tatoeba --limit 30000
```

~87 % unique Japanese sentences (multiple translations merge into a single node).
Data © Tatoeba contributors under [CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).

## Development

Local dev without Docker for the API — just Neo4j in a container:

```bash
docker compose up -d neo4j

# Windows PowerShell
python -m venv venv; .\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv && source venv/bin/activate

pip install -e ".[dev]"
uvicorn app.main:app --reload
```

### Make targets

```bash
make up        # start Neo4j + API, print URLs
make down      # stop everything
make logs      # tail API logs
make seed      # load sample sentences
make import    # import 10 k Tatoeba sentences
make test      # run unit + integration tests
make clean     # stop and wipe all data volumes
```

### Tests

```bash
pytest tests/unit -v                                    # fast, mocked
docker compose --profile test up -d neo4j-test          # ephemeral test DB
pytest tests/integration -v
```

## Project structure

```
app/
├── api/              # FastAPI routers
│   ├── analyze.py    #   POST /analyze, /analyze/cards
│   ├── deps.py       #   dependency injection (singletons)
│   ├── ingest.py     #   POST /ingest
│   ├── sentences.py  #   /sentences CRUD + semantic search
│   ├── stats.py      #   GET /stats
│   └── tokenize.py   #   POST /tokenize
├── db/
│   └── neo4j.py      # async Neo4j driver + lifespan
├── models/
│   ├── ingest.py     # CardIngestPayload, IngestResult
│   ├── sentence.py   # Sentence, AnalysisResult, CardInput, JLPT enums
│   └── token.py      # Token, TokenizeRequest/Response, POS enum
├── services/
│   ├── embeddings.py # sentence-transformers wrapper
│   ├── ingest_service.py  # graph merge logic (idempotent)
│   ├── kanji.py      # kanji extraction helper
│   ├── llm.py        # Groq client — analysis + card generation
│   ├── sentence_service.py  # CRUD + vector search
│   └── tokenizer.py  # Fugashi/MeCab wrapper
├── config.py          # pydantic-settings configuration
└── main.py            # FastAPI app entry point

scripts/
├── import_tatoeba.py  # bulk Tatoeba importer
└── seed_sample.py     # sample data seeder

tests/
├── unit/              # fast, mocked tests
├── integration/       # tests against real Neo4j
└── conftest.py        # shared fixtures
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key for LLM |
| `NEO4J_URI` | — | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | — | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | ✅ | — | Neo4j password |
| `EMBEDDING_MODEL` | — | `intfloat/multilingual-e5-base` | sentence-transformers model |
| `LOG_LEVEL` | — | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `CHROMA_PERSIST_DIR` | — | `./chroma_db` | ChromaDB storage path |

## Related projects

[NihongoCards](https://github.com/blokzz/nihongocards) — Tauri + React desktop
flashcard app with FSRS spaced repetition. Cards approved there are POSTed to
`/ingest` and their returned `word_id` is stored as a pointer to the knowledge graph.

## License

MIT