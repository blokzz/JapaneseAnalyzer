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

class CardExample(BaseModel):
    sentence: str = ""
    furigana: Optional[str] = ""
    reading: Optional[str] = ""
    translation: Optional[str] = ""

class CardInput(BaseModel):
    front: str
    back: Optional[str] = ""
    furigana: Optional[str] = ""
    reading: Optional[str] = ""
    onyomi: Optional[str] = ""
    kunyomi: Optional[str] = ""
    card_type: Literal["vocabulary","phrase","kanji","grammar"] = "vocabulary"
    jlpt: Optional[Literal["N5","N4","N3","N2","N1"]] = None
    meanings: list[str] = []
    examples: list[CardExample] = []
    synonyms: list[str] = []