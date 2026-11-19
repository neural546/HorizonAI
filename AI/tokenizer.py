"""Text tokenizer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

_WORD_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]")


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


class SimpleTokenizer:
    def __init__(self, vocab: Optional[dict] = None, max_length: int = 32):
        self.max_length = int(max_length)
        if vocab is None:
            self.token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        else:
            self.token_to_id = dict(vocab)
        self._rebuild_id_to_token()

    def _rebuild_id_to_token(self) -> None:
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def build_vocab(self, texts: Iterable[str], min_freq: int = 1) -> None:
        counts: dict = {}
        for text in texts:
            for tok in _tokenize_text(text):
                counts[tok] = counts.get(tok, 0) + 1
        for tok in sorted(counts):
            if counts[tok] >= min_freq and tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)
        self._rebuild_id_to_token()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"max_length": self.max_length, "vocab": self.token_to_id}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], max_length=data.get("max_length", 32))

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = _tokenize_text(text)
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
            if add_special_tokens:
                ids[-1] = self.eos_id
        while len(ids) < self.max_length:
            ids.append(self.pad_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special = set(SPECIAL_TOKENS)
        words: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(int(i), UNK_TOKEN)
            if skip_special_tokens and tok in special:
                continue
            words.append(tok)
        out = ""
        for w in words:
            if not out:
                out = w
            elif w in ".,!?;:'\")]":
                out += w
            else:
                out += " " + w
        return out.strip()

    def attention_mask(self, ids: List[int]) -> List[int]:
        return [0 if int(i) == self.pad_id else 1 for i in ids]


def build_default_tokenizer(max_length: int = 32) -> SimpleTokenizer:
    corpus = [
        "a dragon flying above snowy mountains at sunset",
        "a cat walking in the forest",
        "a river flowing through a valley",
        "an astronaut floating in space",
        "a bird flying over the ocean",
        "a city at night with bright lights",
        "a car driving on a highway",
        "a person walking on a beach",
        "fireworks exploding in the sky",
        "a flower blooming in a garden",
        "a robot dancing in the rain",
        "a spaceship traveling through stars",
        "a fish swimming in a coral reef",
        "a train moving through a tunnel",
        "a horse running across a field",
        "a simple colored test scene",
        "a simple test scene",
        "a cat walking",
        "a dragon flying",
    ]
    tok = SimpleTokenizer(max_length=max_length)
    tok.build_vocab(corpus)
    return tok


if __name__ == "__main__":
    tokenizer = build_default_tokenizer()
    sample = "a dragon flying above snowy mountains"
    encoded = tokenizer.encode(sample)
    print("Vocab size:", tokenizer.vocab_size)
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
