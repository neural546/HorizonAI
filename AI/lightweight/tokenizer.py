"""Lightweight tokenizer for the MVP CPU path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from AI.tokenizer import SimpleTokenizer, build_default_tokenizer  # noqa: E402


def build_mvp_tokenizer(max_length: int = 16) -> SimpleTokenizer:
    return build_default_tokenizer(max_length=max_length)
