from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.api.deps import get_ingest_service
from app.models.ingest import CardIngestPayload, IngestResult
from app.services.ingest_service import IngestService

router = APIRouter()

@router.post("", response_model=IngestResult)
async def ingest_card(
    payload: CardIngestPayload,
    service: IngestService = Depends(get_ingest_service),
) -> IngestResult:
    try:
        return await service.ingest(payload)
    except Exception:
        logger.exception(f"Ingest failed for main_text={payload.main_text[:40]}")
        raise HTTPException(500, "Ingest failed")