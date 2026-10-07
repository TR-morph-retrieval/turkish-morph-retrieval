"""Build ``artifacts/seg_{tt,mph}.json``: surface segments for every eligible word of the 600 families.

Run inside the isolated segmenter venv (see README), e.g.
``.venv-seg/bin/python -m experiments.tokenizer_probe.segment_cache --tokenizer mph``
from the repo root. Training and reporting never import the segmenters, only the JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from importlib import metadata
from pathlib import Path

from .data import ARTIFACTS, PURPOSE, SEALED_SHA256, load_items
from .marked_text import item_texts, unique_words
from .segmenters import MORPHEUS_REPO, MorpheusSegmenter, TurkishTokenizerSegmenter, segment_all


def cache_path(name: str, out_dir: Path = ARTIFACTS) -> Path:
    return out_dir / f"seg_{name}.json"


def build(name: str) -> dict:
    items = load_items()
    words = unique_words(item_texts(items))
    if name == "tt":
        segmenter = TurkishTokenizerSegmenter()
        version = {"turkish-tokenizer": metadata.version("turkish-tokenizer")}
    else:
        segmenter = MorpheusSegmenter()
        from huggingface_hub import HfApi

        version = {"repo": MORPHEUS_REPO, "revision": HfApi().model_info(MORPHEUS_REPO).sha}
    segments: dict[str, list[str]] = {}
    reasons: Counter = Counter()
    for word, (pieces, reason) in zip(words, segment_all(segmenter, words)):
        reasons[reason] += 1
        if pieces and len(pieces) > 1:
            segments[word] = pieces
    boundaries = sum(len(p) - 1 for p in segments.values())
    stats = {
        "unique_words": len(words),
        "segmented_words": len(segments),
        "fallback_words": len(words) - reasons["ok"],
        "fallback_rate": round((len(words) - reasons["ok"]) / len(words), 4),
        "reasons": dict(reasons),
        "boundaries": boundaries,
        "boundaries_per_segmented_word": round(boundaries / max(1, len(segments)), 3),
    }
    if name == "tt":
        types: Counter = Counter()
        for word in segments:
            types.update(set(segmenter.piece_types(word)))
        stats["segmented_words_containing_piece_type"] = dict(types)
    return {"purpose": PURPOSE, "tokenizer": name, "version": version, "source_sha256": SEALED_SHA256,
            "stats": stats, "segments": segments}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", choices=("tt", "mph"), required=True)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    payload = build(args.tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    path = cache_path(args.tokenizer, args.out)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=1))
    print("yazıldı:", path)


if __name__ == "__main__":
    main()
