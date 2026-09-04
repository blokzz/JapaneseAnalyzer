import re

KANJI_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def extract_kanji(text: str) -> list[str]:
    seen = set()
    result = []
    for ch in KANJI_PATTERN.findall(text):
        if ch not in seen:
            seen.add(ch)
            result.append(ch)
    return result