from fastapi import APIRouter, Depends

from app.models.token import TokenizeRequest, TokenizeResponse
from app.services.tokenizer import TokenizerService

router = APIRouter()


_tokenizer_singleton: TokenizerService | None = None


def get_tokenizer() -> TokenizerService:
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        _tokenizer_singleton = TokenizerService()
    return _tokenizer_singleton


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