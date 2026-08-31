import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.sentence import JLPTLevel
from app.services.llm import LLMService


@pytest.fixture
def mock_groq_response():
    def _make(content: dict) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(content)
        return response
    return _make


class TestLLMService:
    @pytest.mark.asyncio
    async def test_analyze_parses_response(self, mock_groq_response):
        with patch("app.services.llm.AsyncGroq") as GroqMock:
            client = AsyncMock()
            client.chat.completions.create.return_value = mock_groq_response({
                "level": "N4",
                "grammar_points": ["past tense"],
                "vocabulary": ["食べる"],
                "difficulty_score": 0.3,
                "explanation": "Basic past tense.",
            })
            GroqMock.return_value = client

            service = LLMService()
            result = await service.analyze("昨日ラーメンを食べました")

        assert result.level == JLPTLevel.N4
        assert result.difficulty_score == 0.3
        assert "食べる" in result.vocabulary

    @pytest.mark.asyncio
    async def test_analyze_uses_json_mode(self, mock_groq_response):
        with patch("app.services.llm.AsyncGroq") as GroqMock:
            client = AsyncMock()
            client.chat.completions.create.return_value = mock_groq_response({
                "level": "N5", "grammar_points": [], "vocabulary": [],
                "difficulty_score": 0.1, "explanation": "x",
            })
            GroqMock.return_value = client

            service = LLMService()
            await service.analyze("test")

            call_kwargs = client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}