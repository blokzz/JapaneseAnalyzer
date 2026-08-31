from app.models.sentence import CardInput
import json
from typing import Any

from groq import AsyncGroq, GroqError
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.models.sentence import AnalysisResult, JLPTLevel


ANALYZE_PROMPT = """You are an expert Japanese language teacher specializing in JLPT assessment.

Analyze this Japanese sentence and return a JSON object with these exact fields:
- level: JLPT level ("N5", "N4", "N3", "N2", or "N1") — where N5 is easiest, N1 is hardest and N3 is the best for Intermediate learners.
- grammar_points: array of key grammar patterns colloquial structures and particles used, explained clearly in English (e.g., ["colloquial topic/quote marker って/ってのは", "explanatory form んだ", "conditional たら"]) Be precise and literal (e.g., use "past continuous ~ていた" instead of inventing complex forms like causative-passive if they are not explicitly present).
- vocabulary: array of the most important dictionary-form words in Japanese (e.g., ["食べる", "昨日"])
- difficulty_score: number between 0.0 (very easy) and 1.0 (very hard)
- explanation: 1-2 sentences in English explaining the sentence structure and meaning

Sentence: {text}

Return ONLY valid JSON, no markdown, no extra text."""


CARD_PROMPT_TEMPLATE = """
You are creating high-quality Japanese language learning flashcards. Transform the provided input into a structured JSON array of card objects.

INPUT: "{text}"

### OUTPUT CRITICAL RULES:
1. Output ONLY a valid raw JSON array. Do NOT wrap it in Markdown code blocks (e.g., no ```json). No preamble, no postscript.
2. Maintain clean schema structures. Optional fields with no applicable data MUST be returned as `null` (not empty strings or hallucinated data).

### FIELD SPECIFICATIONS & RULES:

1. `card_type` (required, string):
   - "vocabulary": Single words, compound terms, or verb/adjective forms.
   - "phrase": Multi-word expressions, idioms, or full conversational sentences.
   - "kanji": Single kanji character analysis.
   - "grammar": Grammar patterns or structural particles.
   - "onomatopoeia": Sound-symbolic word or phrase.

2. `front` (required, string): Main Japanese text (in Kanji/Kana as commonly written).
3. `back` (required, string): Concise English translation.

4. `furigana` (optional, string/null):
   - For `vocabulary`/`phrase`/`grammar`: Use bracket notation to map kanji to kana for UI rendering (e.g., "群[むら]がる", "食[た]べる"). If input has no kanji, set to `null` or raw kana string without brackets.
   - Set to `null` for `kanji` type cards.

5. `reading` (optional, string/null): Romaji reading using standard macrons (e.g., "muragaru", "taberu").

6. `onyomi` & `kunyomi` (CRITICAL):
   - ALWAYS set both to `null` for `vocabulary`, `phrase`, and `grammar` card types.
   - ONLY populate these fields if `card_type` is strictly "kanji".
   - `onyomi`: Katakana array or single string (e.g., "カン").
   - `kunyomi`: Hiragana array or single string (e.g., "み.る").

7. `jlpt` (optional, string/null): Estimated level ("N5", "N4", "N3", "N2", "N1") or `null`.
- Estimate the JLPT level strictly based on standard dictionaries (e.g. JMdict/Jisho).
- Kanji complexity must determine the minimum level (e.g. 縁 cannot be N4).
- If uncertain, default to higher levels (e.g. N3/N2 instead of N4/N5).
- If level is unknown or uncertain, aim into the N1 > N2 > N3 range.

8. `meanings` (array of strings): 2-4 primary English definitions or synonyms.

9. `examples` (array of objects): 1 to 3 natural contextual sentences.
   Each object MUST contain:
   - `sentence`: Original sentence in standard Japanese.
   - `furigana`: Sentence with kanji annotated using bracket notation (e.g., "ハエが食[た]べ物[もの]の周[まわ]りに群[むら]がる。").
   - `reading`: Full Romaji reading.
   - `translation`: Natural English translation.

10. `synonyms` & `antonyms` (array of strings): Relevant Japanese words, or empty array `[]`.

### JSON STRUCTURE SCHEMA:
Return a JSON object with a single key "cards" containing an array of card objects:
{{
  "cards": [
    {{
      "card_type": "vocabulary",
      "front": "群がる",
      "back": "to swarm",
      "furigana": "群[むら]がる",
      "reading": "muragaru",
      "onyomi": null,
      "kunyomi": null,
      "jlpt": "N3",
      "meanings": ["to swarm"],
      "examples": [],
      "synonyms": [],
      "antonyms": []
    }}
  ]
}}

INPUT TO PROCESS:
{text}
"""


class LLMService:
    MODEL = "qwen/qwen3.8-27b"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        logger.info(f"Groq client initialized with model {self.MODEL}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GroqError),
        reraise=True,
    )
    async def analyze(self, text: str) -> AnalysisResult:
        response = await self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": ANALYZE_PROMPT.format(text=text)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from LLM")

        data: dict[str, Any] = json.loads(raw)
        logger.debug(f"LLM raw response: {data}")

        return AnalysisResult(
            sentence=text,
            level=JLPTLevel(data["level"]),
            grammar_points=data.get("grammar_points", []),
            vocabulary=data.get("vocabulary", []),
            difficulty_score=float(data["difficulty_score"]),
            explanation=data["explanation"],
        )
    async def create_cards(self, text: str) -> list[CardInput]:
        response = await self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "user", "content": CARD_PROMPT_TEMPLATE.format(text=text)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from LLM")

        data: dict[str, Any] = json.loads(raw)
        logger.debug(f"LLM raw response: {data}")

        if "cards" in data:
            cards_raw = data["cards"]
            if not isinstance(cards_raw, list):
                cards_raw = [cards_raw]
        elif "front" in data:
            cards_raw = [data]
        else:
            cards_raw = []

        return [CardInput(**item) for item in cards_raw]