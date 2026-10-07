"""Surface-form morpheme segmenters (TurkishTokenizer, Morpheus) behind one interface.

Both return the segments of the original-case word, or ``None`` plus a reason when the tool's
output cannot be mapped back onto the surface word. Callers fall back to "no boundary".
Run in the isolated segmenter venv: they import ``turkish_tokenizer`` / the Morpheus clone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from .data import ROOT  # noqa: F401  (puts the repo root first on sys.path)
from test.validators import tr_lower  # noqa: E402

VENDOR = Path(__file__).resolve().parent / ".vendor" / "TurkishMorpheus"
MORPHEUS_REPO = "lonewolflab/Morpheus-TR-50K"


class TurkishTokenizerSegmenter:
    """Boundaries of the Rust ``turkish-tokenizer`` (roots, canonical affixes, BPE fallback).

    Its tokens are canonical (``-lar`` for ``-ler``, ``kitap`` for ``kitab``), so surface cut
    points come from decoding token prefixes: the prefix decode may differ from the surface in
    its last character only (lenition), never earlier.
    """

    name = "tt"

    def __init__(self) -> None:
        import turkish_tokenizer

        self._tok = turkish_tokenizer.TurkishTokenizer()

    def _decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids).strip()

    def segment(self, word: str) -> tuple[list[str] | None, str]:
        lowered = tr_lower(word)
        if len(lowered) != len(word):
            return None, "length_changed"
        pieces = [p for p in self._tok.tokenize_text(lowered) if p["token"] != "<uppercase>"]
        ids = [p["id"] for p in pieces]
        if not ids:
            return None, "empty"
        if self._decode(ids) != lowered:
            return None, "decode_mismatch"
        cuts: list[int] = []
        for end in range(1, len(ids)):
            prefix = self._decode(ids[:end])
            cut = len(prefix)
            if prefix != lowered[:cut] and prefix[:-1] != lowered[:cut - 1]:
                return None, "prefix_mismatch"
            if cut <= (cuts[-1] if cuts else 0) or cut >= len(lowered):
                return None, "bad_cut"
            cuts.append(cut)
        bounds = [0] + cuts + [len(word)]
        return [word[a:b] for a, b in zip(bounds, bounds[1:])], "ok"

    def piece_types(self, word: str) -> list[str]:
        lowered = tr_lower(word)
        return [p["type"] for p in self._tok.tokenize_text(lowered) if p["token"] != "<uppercase>"]


class MorpheusSegmenter:
    """Boundaries of the Morpheus neural morpheme-boundary model (lossless by construction)."""

    name = "mph"

    def __init__(self, device: str = "cpu") -> None:
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        from src.model_development.tokenization.morpheus_tokenizer import MorpheusTokenizer

        self._tok = MorpheusTokenizer.from_pretrained(MORPHEUS_REPO, device=device)

    def segment_many(self, words: list[str]) -> list[tuple[list[str] | None, str]]:
        results = []
        for word, segments in zip(words, self._tok.segment_words_batched(words)):
            if "".join(segments) != word:
                results.append((None, "not_lossless"))
            else:
                results.append((list(segments), "ok"))
        return results


def segment_all(segmenter, words: Iterable[str]) -> list[tuple[list[str] | None, str]]:
    words = list(words)
    if hasattr(segmenter, "segment_many"):
        return segmenter.segment_many(words)
    return [segmenter.segment(word) for word in words]
