import pytest
from pydantic import ValidationError

from app.models.sentence import AnalysisResult, JLPTLevel, SentenceCreate


class TestSentenceCreate:
    def test_valid_sentence(self):
        s = SentenceCreate(text="私は学生です", translation="I am a student")
        assert s.text == "私は学生です"
        assert s.translation == "I am a student"
        assert s.source is None

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError, match="at least 1 character"):
            SentenceCreate(text="")

    def test_whitespace_stripped(self):
        s = SentenceCreate(text="  私は学生です  ")
        assert s.text == "私は学生です"

    def test_too_long_text_rejected(self):
        with pytest.raises(ValidationError, match="at most 500"):
            SentenceCreate(text="あ" * 501)


class TestAnalysisResult:
    def test_valid_result(self):
        r = AnalysisResult(
            sentence="test",
            level=JLPTLevel.N3,
            grammar_points=["particle は"],
            vocabulary=["test"],
            difficulty_score=0.5,
            explanation="ok",
        )
        assert r.level == JLPTLevel.N3

    def test_difficulty_score_range(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                sentence="x", level=JLPTLevel.N5,
                difficulty_score=1.5, 
                explanation="x",
            )

    def test_invalid_jlpt_level(self):
        with pytest.raises(ValidationError):
            AnalysisResult(
                sentence="x", level="N9", 
                difficulty_score=0.5, explanation="x",
            )