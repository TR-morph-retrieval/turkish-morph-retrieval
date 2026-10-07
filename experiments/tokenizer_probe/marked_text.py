"""Morpheme-boundary marking for the XLM-R tokenizer, plus the experiment arms.

A boundary is a private-use character inserted between morphemes of a word. The patched
tokenizer (see ``patch_tokenizer``) treats it as a hard pre-token split that emits nothing, so
each morpheme is encoded separately by the original Unigram model while the embedding matrix,
vocabulary and ids stay untouched. Text without the marker tokenizes exactly as before.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .data import ARTIFACTS

# Verified against the mE5-large tokenizer: absent from the vocab, survives the normalizer.
MARKER = ""
WORD_RE = re.compile(r"[^\W\d_]+")
APOSTROPHES = "'’ʼ"

# shift_*: every boundary moved by one character (same count and granularity, morphology broken).
# rand_*: same boundary count at random positions (much more disruptive; zero-shot diagnostic only).
ARMS = ("base", "tt", "mph", "shift_tt", "shift_mph", "rand_tt", "rand_mph")
SEGMENTED_ARMS = {"tt": "tt", "mph": "mph", "shift_tt": "tt", "shift_mph": "mph", "rand_tt": "tt", "rand_mph": "mph"}


def load_caches(directory: Path = ARTIFACTS) -> dict[str, dict[str, list[str]]]:
    """Segment caches written by ``segment_cache`` (``seg_tt.json``, ``seg_mph.json``)."""
    return {name: json.loads((Path(directory) / f"seg_{name}.json").read_text(encoding="utf-8"))["segments"]
            for name in ("tt", "mph")}


def eligible_words(text: str) -> Iterable[re.Match[str]]:
    """Letter runs that may be segmented: suffixes glued on by an apostrophe stay untouched."""
    for match in WORD_RE.finditer(text):
        start = match.start()
        if start > 0 and text[start - 1] in APOSTROPHES:
            continue
        yield match


def unique_words(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    for text in texts:
        for match in eligible_words(text):
            seen.add(match.group())
    return sorted(seen)


def mark_text(text: str, segments: Mapping[str, list[str]]) -> str:
    """Insert ``MARKER`` between the cached segments of every eligible word."""
    out: list[str] = []
    last = 0
    for match in eligible_words(text):
        pieces = segments.get(match.group())
        if not pieces or len(pieces) < 2:
            continue
        out.append(text[last:match.start()])
        out.append(MARKER.join(pieces))
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def random_segments(word: str, n_boundaries: int, seed: int) -> list[str]:
    """``n_boundaries`` random cut points inside ``word`` (deterministic per seed and word)."""
    if n_boundaries <= 0 or len(word) < 2:
        return [word]
    n_boundaries = min(n_boundaries, len(word) - 1)
    digest = hashlib.sha256(f"{seed}:{word}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    cuts = sorted(rng.sample(range(1, len(word)), n_boundaries))
    return [word[a:b] for a, b in zip([0] + cuts, cuts + [len(word)])]


def shifted_segments(word: str, segments: list[str], seed: int) -> list[str]:
    """Move each boundary one character left or right (random sign), never onto another boundary."""
    cuts, total = [], 0
    for piece in segments[:-1]:
        total += len(piece)
        cuts.append(total)
    digest = hashlib.sha256(f"{seed}:shift:{word}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    shifted: list[int] = []
    for cut in cuts:
        options = [cut - 1, cut + 1]
        rng.shuffle(options)
        shifted.append(next((o for o in options if 0 < o < len(word) and o not in cuts and o not in shifted), cut))
    bounds = [0] + sorted(shifted) + [len(word)]
    return [word[a:b] for a, b in zip(bounds, bounds[1:])]


def shifted_cache(reference: Mapping[str, list[str]], seed: int = 42) -> dict[str, list[str]]:
    return {word: shifted_segments(word, pieces, seed) for word, pieces in reference.items() if len(pieces) > 1}


def random_cache(reference: Mapping[str, list[str]], words: Iterable[str], seed: int = 42) -> dict[str, list[str]]:
    """Control cache: same boundary count per word as ``reference``, random positions."""
    out = {}
    for word in words:
        count = len(reference.get(word, [word])) - 1
        if count > 0:
            out[word] = random_segments(word, count, seed)
    return out


def arm_cache(arm: str, caches: Mapping[str, Mapping[str, list[str]]], words: Iterable[str], seed: int = 42):
    """Segment mapping for an arm, or ``None`` for the untouched baseline."""
    if arm == "base":
        return None
    reference = caches[SEGMENTED_ARMS[arm]]
    if arm.startswith("rand_"):
        return random_cache(reference, words, seed)
    if arm.startswith("shift_"):
        return shifted_cache(reference, seed)
    return reference


def mark_items(items: list[dict[str, Any]], segments: Mapping[str, list[str]] | None) -> list[dict[str, Any]]:
    """Deep copy of the families with marked ``query`` and candidate ``text`` (ids unchanged)."""
    marked = copy.deepcopy(items)
    if segments is None:
        return marked
    for item in marked:
        item["query"] = mark_text(item["query"], segments)
        for candidate in item["candidates"]:
            candidate["text"] = mark_text(candidate["text"], segments)
    return marked


def item_texts(items: list[dict[str, Any]]) -> list[str]:
    return [item["query"] for item in items] + [c["text"] for item in items for c in item["candidates"]]


def patch_tokenizer(tokenizer, marker: str = MARKER):
    """Make ``marker`` a silent hard split after the existing pre-tokenizer (idempotent)."""
    from tokenizers import Regex, pre_tokenizers

    if getattr(tokenizer, "_probe_marker_patched", False):
        return tokenizer
    backend = tokenizer.backend_tokenizer
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        backend.pre_tokenizer,
        pre_tokenizers.Split(Regex(re.escape(marker)), behavior="removed"),
    ])
    tokenizer._probe_marker_patched = True
    return tokenizer


def load_patched_tokenizer(name: str = "intfloat/multilingual-e5-large"):
    """The mE5-large tokenizer with the boundary marker enabled (local cache first)."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(name)
    return patch_tokenizer(tokenizer)
