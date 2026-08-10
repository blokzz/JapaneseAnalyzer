import fugashi
from loguru import logger

from app.models.token import PartOfSpeech, Token


POS_MAP: dict[str, PartOfSpeech] = {
    "名詞": PartOfSpeech.NOUN,
    "動詞": PartOfSpeech.VERB,
    "形容詞": PartOfSpeech.ADJECTIVE,
    "形状詞": PartOfSpeech.ADJECTIVE,       # な-adjectives
    "副詞": PartOfSpeech.ADVERB,
    "助詞": PartOfSpeech.PARTICLE,
    "助動詞": PartOfSpeech.AUXILIARY,
    "代名詞": PartOfSpeech.PRONOUN,
    "接続詞": PartOfSpeech.CONJUNCTION,
    "感動詞": PartOfSpeech.INTERJECTION,
    "接頭辞": PartOfSpeech.PREFIX,
    "接尾辞": PartOfSpeech.SUFFIX,
    "補助記号": PartOfSpeech.SYMBOL,
    "記号": PartOfSpeech.SYMBOL,
}

SKIP_POS = {PartOfSpeech.SYMBOL}


class TokenizerService:
    def __init__(self) -> None:
        self._tagger = fugashi.Tagger()  # type: ignore[attr-defined]
        logger.info("Fugashi tokenizer initialized")

    def tokenize(self, text: str, skip_symbols: bool = True) -> list[Token]:
        tokens: list[Token] = []
        for word in self._tagger(text):
            token = self._to_token(word)
            if skip_symbols and token.pos in SKIP_POS:
                continue
            tokens.append(token)
        return tokens

    def _to_token(self, word) -> Token:
        feature = word.feature
        pos_ja = feature.pos1 or "その他"
        pos = POS_MAP.get(pos_ja, PartOfSpeech.OTHER)

        lemma = getattr(feature, "lemma", None) or word.surface
        reading = getattr(feature, "kana", None) or getattr(feature, "pron", None)
        pos_detail = f"{feature.pos1}-{feature.pos2}" if feature.pos2 else feature.pos1

        return Token(
            surface=word.surface,
            lemma=lemma,
            reading=reading,
            pos=pos,
            pos_detail=pos_detail,
        )