from fastapi import FastAPI
from app.api import analyze, sentences, stats, tokenize
from app.db.neo4j import lifespan
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Japanese RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost"],
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(stats.router, prefix="/stats", tags=["stats"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(sentences.router, prefix="/sentences", tags=["sentences"])
app.include_router(tokenize.router, prefix="/tokenize", tags=["tokenize"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


