
from fastapi._compat.v2 import ValidationError
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


CARD_SYSTEM_PROMPT = """You are a Japanese language teacher creating high-quality flashcards.

You output ONLY valid JSON matching the exact schema below. No preamble, no comments.

# Schema

Always return a JSON object with a single "cards" key containing an array,
even when generating a single card:

{
  "cards": [
    {
      "card_type": "vocabulary" | "phrase" | "kanji" | "grammar",
      "front":    string,        // Japanese text as commonly written (kanji + kana)
      "back":     string,        // Concise English translation
      "furigana": string | null, // Anki notation, e.g. "群[むら]がる"
      "reading":  string | null, // Romaji with macrons, e.g. "muragaru"
      "onyomi":   string | null, // Katakana; ONLY for card_type="kanji"
      "kunyomi":  string | null, // Hiragana; ONLY for card_type="kanji"
      "jlpt":     "N5" | "N4" | "N3" | "N2" | "N1" | null,
      "meanings": [string],      // 2-4 English definitions
      "examples": [
        {
          "sentence":    string, // Natural Japanese sentence
          "furigana":    string, // Anki notation as above
          "reading":     string, // Full romaji
          "translation": string  // Natural English
        }
      ],
      "synonyms": [string]       // Japanese words, or []
    }
  ]
}

# Rules

- `onyomi` and `kunyomi` MUST be null unless `card_type="kanji"`.
- `furigana` MUST be null when `card_type="kanji"`.
- For inputs with no kanji, `furigana` is null.
- Missing optional data is `null`, never an empty string or made-up value.
- Generate 1-3 examples per card. Fewer is fine — better than fabricated.

# JLPT level (be strict)

- Base level on kanji complexity + grammar difficulty per JMdict/Jisho conventions.
- Kanji complexity is a floor: 縁 cannot be N5 even if word is basic.
- When uncertain, prefer HIGHER levels: N3 > N4, N2 > N3, N1 > N2.
- If truly unknown, use N1.

# Card type selection

- **vocabulary**: single word or short compound (食べる, 学生, 一生懸命)
- **phrase**: multi-word expression or idiom (お疲れ様でした, 猫の手も借りたい)
- **kanji**: single kanji character analysis (漢, 縁)
- **grammar**: pattern or structural particle (〜てしまう, 〜わけではない)
- **onomatopoeia**: sound-symbolic word or phrase (ワンワン, ドキドキ)
"""

CARD_SYSTEM_PROMPT_WITH_EXAMPLES = CARD_SYSTEM_PROMPT + """

# Examples

Input: 食べる
Output: {"cards":[{"card_type":"vocabulary","front":"食べる","back":"to eat",
"furigana":"食[た]べる","reading":"taberu","onyomi":null,"kunyomi":null,
"jlpt":"N5","meanings":["to eat","to consume"],
"examples":[{"sentence":"寿司を食べます。","furigana":"寿司[すし]を食[た]べます。",
"reading":"Sushi wo tabemasu.","translation":"I eat sushi."}],
"synonyms":["食う","召し上がる"]}]}

Input: 縁
Output: {"cards":[{"card_type":"kanji","front":"縁","back":"edge, connection, fate",
"furigana":null,"reading":null,"onyomi":"エン","kunyomi":"ふち, ゆかり",
"jlpt":"N1","meanings":["edge","border","connection","karmic bond"],
"examples":[{"sentence":"縁がある。","furigana":"縁[えん]がある。",
"reading":"En ga aru.","translation":"There is a connection/fate."}],
"synonyms":["因縁","縁故"]}]}

Input: 〜てしまう
Output: {"cards":[{"card_type":"grammar","front":"〜てしまう","back":"to do completely / regretfully",
"furigana":null,"reading":"~te shimau","onyomi":null,"kunyomi":null,
"jlpt":"N4","meanings":["do completely","end up doing (regret)"],
"examples":[{"sentence":"宿題を忘れてしまいました。",
"furigana":"宿題[しゅくだい]を忘[わす]れてしまいました。",
"reading":"Shukudai wo wasurete shimaimashita.",
"translation":"I ended up forgetting my homework."}],
"synonyms":[]}]}
"""

CARD_USER_TEMPLATE = "Input: {text}"

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
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(GroqError),
        reraise=True,
    )
    async def create_cards(self, text: str) -> list[CardInput]:
        response = await self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": CARD_SYSTEM_PROMPT_WITH_EXAMPLES},
                {"role": "user", "content": CARD_USER_TEMPLATE.format(text=text)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from LLM")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {raw[:200]}")
            raise ValueError(f"LLM response is not valid JSON: {e}") from e

        cards_raw = self._extract_cards(data)

        cards: list[CardInput] = []
        for i, item in enumerate(cards_raw):
            try:
                cards.append(CardInput(**item))
            except ValidationError as e:
                logger.warning(f"Card {i} failed validation, skipping: {e.errors()}")

        if not cards:
            raise ValueError("No valid cards in LLM response")

        return cards

    @staticmethod
    def _extract_cards(data: dict) -> list[dict]:
        if "cards" in data:
            cards = data["cards"]
            return cards if isinstance(cards, list) else [cards]
        if "front" in data:
            return [data]
        return []