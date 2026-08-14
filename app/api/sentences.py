from app.services.llm import LLMService
from app.api.deps import get_neo4j_driver, get_llm_service
from fastapi import APIRouter, Depends, HTTPException, Query, status
from neo4j import AsyncDriver

from app.models.sentence import Sentence, SentenceCreate
from app.services.sentence_service import SentenceService
from app.api.tokenize import get_tokenizer
from app.services.tokenizer import TokenizerService
from app.api.deps import get_embedding_service
from app.services.embeddings import EmbeddingService
from typing import Annotated
from fastapi import Query

from app.models.sentence import JLPTLevel, SimilarSentence

router = APIRouter()


def get_service(
    driver: AsyncDriver = Depends(get_neo4j_driver),
    tokenizer: TokenizerService = Depends(get_tokenizer),
    llm: LLMService = Depends(get_llm_service),
    embeddings: EmbeddingService = Depends(get_embedding_service),
) -> SentenceService:
    return SentenceService(driver, tokenizer, llm, embeddings)

@router.get("/similar-with-word", response_model=list[SimilarSentence])
async def similar_with_word(
    q: str, word: str, limit: int = 10,
    service: SentenceService = Depends(get_service),
):
    return await service.find_similar_using_word(q, word, limit)

@router.get("/similar", response_model=list[SimilarSentence])
async def find_similar_sentences(
    q: Annotated[str, Query(min_length=1, max_length=500, description="Query text (any language)")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
    level: JLPTLevel | None = None,
    service: SentenceService = Depends(get_service),
) -> list[SimilarSentence]:
    return await service.find_similar(q, limit=limit, min_score=min_score, level=level)

@router.post("", response_model=Sentence, status_code=status.HTTP_201_CREATED)
async def create_sentence(
    payload: SentenceCreate,
    analyze: bool = Query(default=True),
    service: SentenceService = Depends(get_service),
) -> Sentence:
    return await service.create(payload, analyze=analyze)


@router.get("/{sentence_id}", response_model=Sentence)
async def get_sentence(
    sentence_id: str,
    service: SentenceService = Depends(get_service),
) -> Sentence:
    sentence = await service.get_by_id(sentence_id)
    if not sentence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sentence {sentence_id} not found",
        )
    return sentence


@router.get("", response_model=list[Sentence])
async def list_sentences(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: SentenceService = Depends(get_service),
) -> list[Sentence]:
    return await service.list_all(limit=limit, offset=offset)