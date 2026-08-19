from datetime import datetime
from enum import Enum
from typing import Annotated

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
    source: str | None = Field(default=None, description="np. 'tatoeba', 'manual'")


class Sentence(SentenceCreate):
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