from fastapi import FastAPI
from app.api import sentences, tokenize

from app.config import get_settings
from app.models.sentence import SentenceCreate
from app.db.neo4j import lifespan

app = FastAPI(title="Japanese RAG", version="0.1.0", lifespan=lifespan)
app.include_router(sentences.router, prefix="/sentences", tags=["sentences"])
app.include_router(tokenize.router, prefix="/tokenize", tags=["tokenize"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(sentence: SentenceCreate) -> dict:
    settings = get_settings()
    return {
        "received": sentence.model_dump(),
        "embedding_model": settings.embedding_model,
    }