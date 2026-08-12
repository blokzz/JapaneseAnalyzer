from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.api.deps import get_llm_service
from app.models.sentence import AnalysisResult, SentenceCreate
from app.services.llm import LLMService

router = APIRouter()


@router.post("", response_model=AnalysisResult)
async def analyze_sentence(
    payload: SentenceCreate,
    llm: LLMService = Depends(get_llm_service),
) -> AnalysisResult:
    try:
        return await llm.analyze(payload.text)
    except Exception as e:
        logger.exception(f"LLM analysis failed for: {payload.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM analysis failed: {e}",
        )