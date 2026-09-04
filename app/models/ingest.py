
from app.models.sentence import JapaneseText, JLPTLevel
from pydantic import BaseModel, Field

class IngestResult(BaseModel):
    word_id: str 

class ExampleIngest(BaseModel):
    sentence: JapaneseText
    translation: str | None = None

class CardIngestPayload(BaseModel):
    main_text: JapaneseText
    translation: str | None = None
    jlpt: JLPTLevel | None = None
    examples: list[ExampleIngest] = Field(default_factory=list)