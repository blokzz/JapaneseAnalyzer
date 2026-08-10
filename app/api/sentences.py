from fastapi import APIRouter, Depends, HTTPException, Query, status
from neo4j import AsyncDriver

from app.api.deps import get_neo4j_driver
from app.models.sentence import Sentence, SentenceCreate
from app.services.sentence_service import SentenceService
from app.api.tokenize import get_tokenizer
from app.services.tokenizer import TokenizerService

router = APIRouter()

def get_service(
    driver: AsyncDriver = Depends(get_neo4j_driver),
    tokenizer: TokenizerService = Depends(get_tokenizer),
) -> SentenceService:
    return SentenceService(driver, tokenizer)

@router.post("", response_model=Sentence, status_code=status.HTTP_201_CREATED)
async def create_sentence(
    payload: SentenceCreate,
    service: SentenceService = Depends(get_service),
) -> Sentence:
    return await service.create(payload)


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