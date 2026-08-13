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
- level: JLPT level ("N5", "N4", "N3", "N2", or "N1") — where N5 is easiest, N1 is hardest
- grammar_points: array of key grammar patterns and colloquial structures used, explained clearly in English (e.g., ["colloquial topic/quote marker って/ってのは", "explanatory form んだ", "conditional たら"])
- vocabulary: array of the most important dictionary-form words in Japanese (e.g., ["食べる", "昨日"])
- difficulty_score: number between 0.0 (very easy) and 1.0 (very hard)
- explanation: 1-2 sentences in English explaining the sentence structure and meaning

Sentence: {text}

Return ONLY valid JSON, no markdown, no extra text."""


class LLMService:
    MODEL = "llama-3.3-70b-versatile"

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