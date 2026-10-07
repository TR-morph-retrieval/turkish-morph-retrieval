"""Sealed-600 loading and training-row construction for the tokenizer probe.

Read-only on ``test/``. Everything derived from the 600 families is tagged ``PURPOSE`` and is
not a sealed-test result: the families are used for training and selection here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# The stdlib also ships a ``test`` package; the repo root must win.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.taxonomy import MORPH_HARD_SUBTYPES, SEMANTIC_HARD_SUBTYPES  # noqa: E402

SEALED_PATH = ROOT / "test/data/morph_test_600_sealed.json"
SEALED_SHA256 = "a38b1e4f5930da778f76b77682085ab89ff8aab678cc8b45342f675fa3063b90"
PURPOSE = "tokenizer_probe_only"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
RUNS = Path(__file__).resolve().parent / "runs"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# UTF-8 bytes that were decoded as cp1252/latin-1 ("sayÄ±m"); three sealed families carry them.
_MOJIBAKE = re.compile(
    "[\u00c3\u00c5\u00c4][\u0080-\u00ff\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc"
    "\u2013\u2014\u2018-\u201e\u2020-\u2022\u2026\u2030\u2039\u203a\u20ac\u2122]"
)


def has_mojibake(text: str) -> bool:
    return bool(_MOJIBAKE.search(text))


def repair_text(text: str) -> str:
    """Undo a cp1252/latin-1 mis-decoding; return the text unchanged if it is not such a case."""
    if not has_mojibake(text):
        return text
    for encoding in ("cp1252", "latin-1"):
        try:
            return text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
    return text


def _repair_value(value: Any, counter: list[int]) -> Any:
    if isinstance(value, str):
        fixed = repair_text(value)
        counter[0] += fixed != value
        return fixed
    if isinstance(value, list):
        return [_repair_value(v, counter) for v in value]
    if isinstance(value, dict):
        return {k: _repair_value(v, counter) for k, v in value.items()}
    return value


def load_sealed(path: Path = SEALED_PATH, verify: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the 600 families (mojibake repaired in memory) and a report of what was repaired.

    The file is hash-pinned and never rewritten; the repair only touches this process's copy.
    """
    raw = Path(path).read_bytes()
    if verify and hashlib.sha256(raw).hexdigest() != SEALED_SHA256:
        raise ValueError(f"{path} sha256 pin'iyle eşleşmiyor; dosya değişmiş olabilir.")
    items = json.loads(raw)["items"]
    if len(items) != 600:
        raise ValueError(f"600 family bekleniyordu, {len(items)} bulundu.")
    counter = [0]
    repaired_families = []
    for index, item in enumerate(items):
        before = counter[0]
        items[index] = _repair_value(item, counter)
        if counter[0] != before:
            repaired_families.append(item["family_id"])
    seen: set[str] = set()
    for item in items:
        family_id = item["family_id"]
        if family_id in seen:
            raise ValueError(f"Tekrarlı family_id: {family_id}")
        seen.add(family_id)
        roles = [candidate["role"] for candidate in item["candidates"]]
        if (roles.count("positive"), roles.count("hard_negative"), roles.count("easy_negative")) != (1, 8, 2):
            raise ValueError(f"1+8+2 yapısı bozuk: {family_id}")
        positive = next(c for c in item["candidates"] if c["role"] == "positive")
        if item["gold_id"] != positive["id"]:
            raise ValueError(f"gold_id positive ile eşleşmiyor: {family_id}")
        texts = [item["query"]] + [c["text"] for c in item["candidates"]]
        if any(has_mojibake(t) for t in texts):
            raise ValueError(f"Onarılamayan bozuk karakter: {family_id}")
    report = {"mojibake_strings_repaired": counter[0], "families": sorted(repaired_families)}
    return items, report


def load_items(path: Path = SEALED_PATH, verify: bool = True) -> list[dict[str, Any]]:
    return load_sealed(path, verify)[0]


def _ranked(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda c: hashlib.sha256(f"{seed}:{c['id']}".encode()).hexdigest())


def training_row(item: dict[str, Any], seed: int = 42) -> dict[str, str]:
    """One family as the C5 notebook's ``morph_1, morph_2, semantic_1`` row (2 morph + 1 semantic hard).

    Selection is a deterministic hash ranking, so every arm trains on identical negatives.
    """
    candidates = item["candidates"]
    positive = next(c for c in candidates if c["role"] == "positive")
    hards = [c for c in candidates if c["role"] == "hard_negative"]
    morph = _ranked([c for c in hards if c["subtype"] in MORPH_HARD_SUBTYPES], seed)[:2]
    semantic = _ranked([c for c in hards if c["subtype"] in SEMANTIC_HARD_SUBTYPES], seed)[:1]
    taken = {c["id"] for c in morph + semantic}
    spare = _ranked([c for c in hards if c["id"] not in taken], seed)
    while len(morph) < 2:
        morph.append(spare.pop(0))
    if not semantic:
        semantic.append(spare.pop(0))
    return {
        "family_id": item["family_id"],
        "query": item["query"].strip(),
        "positive": positive["text"].strip(),
        "morph_1": morph[0]["text"].strip(),
        "morph_2": morph[1]["text"].strip(),
        "semantic_1": semantic[0]["text"].strip(),
    }


def training_rows(items: list[dict[str, Any]], seed: int = 42) -> list[dict[str, str]]:
    return [training_row(item, seed) for item in items]
