from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class JLPTLevel(str, Enum):
    N5 = "N5"
    N4 = "N4"
    N3 = "N3"
    N2 = "N2"
    N1 = "N1"


JapaneseText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=500, strip_whitespace=True),
]


class SentenceCreate(BaseModel):
    text: JapaneseText
    translation: str | None = None
    source: str | None = None


class Sentence(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    text: str
    translations: list[str] = Field(default_factory=list)
    source: str | None = None
    level: JLPTLevel | None = None
    created_at: datetime

class SimilarSentence(BaseModel):
    id: str
    text: str
    translations: list[str] = Field(default_factory=list)   
    level: JLPTLevel | None
    score: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    sentence: str
    level: JLPTLevel
    grammar_points: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    difficulty_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class CardType(str, Enum):
    VOCABULARY = "vocabulary"
    PHRASE = "phrase"
    KANJI = "kanji"
    GRAMMAR = "grammar"
    ONOMATOPOEIA = "onomatopoeia"

class CardExample(BaseModel):
    sentence: str = ""
    furigana: str | None = None
    reading: str | None = None
    translation: str | None = None

class CardInput(BaseModel):
    front: str
    back: str | None = None
    furigana: str | None = None
    reading: str | None = None
    onyomi: str | None = None
    kunyomi: str | None = None
    card_type: CardType = CardType.VOCABULARY   
    jlpt: JLPTLevel | None = None                
    meanings: list[str] = Field(default_factory=list)
    examples: list[CardExample] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)

class CardGenerateRequest(BaseModel):
    text: JapaneseText