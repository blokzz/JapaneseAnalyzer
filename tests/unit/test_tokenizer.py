import pytest

from app.models.token import PartOfSpeech
from app.services.tokenizer import TokenizerService


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerService:
    return TokenizerService()


class TestTokenizer:
    def test_basic_sentence(self, tokenizer):
        tokens = tokenizer.tokenize("私は学生です")
        lemmas = [t.lemma for t in tokens]
        assert "私" in lemmas
        assert "学生" in lemmas

    def test_verb_lemma_form(self, tokenizer):
        tokens = tokenizer.tokenize("食べました")
        verbs = [t for t in tokens if t.pos == PartOfSpeech.VERB]
        assert any(t.lemma == "食べる" for t in verbs)

    def test_symbols_skipped_by_default(self, tokenizer):
        tokens = tokenizer.tokenize("私は学生です。")
        surfaces = [t.surface for t in tokens]
        assert "。" not in surfaces

    def test_symbols_included_when_disabled(self, tokenizer):
        tokens = tokenizer.tokenize("私は学生です。", skip_symbols=False)
        surfaces = [t.surface for t in tokens]
        assert "。" in surfaces

    def test_particle_detection(self, tokenizer):
        tokens = tokenizer.tokenize("私は学生です")
        particles = [t for t in tokens if t.pos == PartOfSpeech.PARTICLE]
        assert any(t.surface == "は" for t in particles)

    def test_empty_string(self, tokenizer):
        assert tokenizer.tokenize("") == []