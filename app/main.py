from fastapi import FastAPI
from app.config import get_settings
from app.models.sentence import SentenceCreate

app = FastAPI(title="Japanese RAG", version="0.1.0")


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