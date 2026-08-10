from enum import Enum

from pydantic import BaseModel, Field


class PartOfSpeech(str, Enum):
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PARTICLE = "particle"
    AUXILIARY = "auxiliary"
    PRONOUN = "pronoun"
    CONJUNCTION = "conjunction"
    INTERJECTION = "interjection"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    SYMBOL = "symbol"
    OTHER = "other"


class Token(BaseModel):
    surface: str = Field(description="Forma w tekście, np. 食べました")
    lemma: str = Field(description="Forma słownikowa, np. 食べる")
    reading: str | None = Field(default=None, description="Czytanie w katakanie, np. タベマシタ")
    pos: PartOfSpeech = Field(description="Część mowy")
    pos_detail: str | None = Field(default=None, description="Szczegół z MeCaba, np. '動詞-一般'")


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class TokenizeResponse(BaseModel):
    text: str
    tokens: list[Token]
    token_count: int