from fastapi import APIRouter, Depends

from app.models.token import TokenizeRequest, TokenizeResponse
from app.services.tokenizer import TokenizerService
from app.api.deps import get_tokenizer

router = APIRouter()

@router.post("", response_model=TokenizeResponse)
async def tokenize_text(
    payload: TokenizeRequest,
    tokenizer: TokenizerService = Depends(get_tokenizer),
) -> TokenizeResponse:
    tokens = tokenizer.tokenize(payload.text)
    return TokenizeResponse(
        text=payload.text,
        tokens=tokens,
        token_count=len(tokens),
    )