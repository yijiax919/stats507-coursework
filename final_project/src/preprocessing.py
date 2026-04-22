from __future__ import annotations

import re

URL_RE = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
USER_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
SPACE_RE = re.compile(r"\s+")


def normalize_tweet(text: str) -> str:
    """Light normalization that keeps most lexical signal intact."""
    text = str(text).lower()
    text = URL_RE.sub(" URL ", text)
    text = USER_RE.sub(" USER ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text
