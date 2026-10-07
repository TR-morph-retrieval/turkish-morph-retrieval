"""Local, GPU-free audit of what each arm does to the tokenization of the 600 families.

Reports, per arm: share of words whose tokenization changes, length inflation, truncation at the
training ``max_seq_length`` and how clean the contrast is between the positive's and the morph
hard negatives' critical words. Needs ``tokenizers`` + ``transformers`` (no torch).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from test.taxonomy import MORPH_HARD_SUBTYPES

from .data import ARTIFACTS, PURPOSE, load_items
from .marked_text import (ARMS, MARKER, arm_cache, item_texts, load_caches, load_patched_tokenizer, mark_items,
                          unique_words)

MAX_SEQ_LENGTH = 256


def cut_positions(pieces: list[str]) -> set[int]:
    lengths = [len(p.replace("▁", "")) for p in pieces]
    lengths = [n for n in lengths if n]
    total, cuts = 0, set()
    for n in lengths[:-1]:
        total += n
        cuts.add(total)
    return cuts


def lookup(segments: dict[str, list[str]], word: str) -> list[str] | None:
    for form in (word, word.capitalize(), word.lower(), word.upper()):
        if form in segments:
            return segments[form]
    return None


def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def audit(directory: Path = ARTIFACTS) -> dict[str, Any]:
    items = load_items()
    words = unique_words(item_texts(items))
    caches = load_caches(directory)
    tokenizer = load_patched_tokenizer()
    base_pieces = {w: tokenizer.tokenize(w) for w in words}

    critical_pairs = []
    for item in items:
        positive = next(c for c in item["candidates"] if c["role"] == "positive")
        for candidate in item["candidates"]:
            if candidate["role"] == "hard_negative" and candidate["subtype"] in MORPH_HARD_SUBTYPES \
                    and candidate["critical_word"] != positive["critical_word"]:
                critical_pairs.append((item["macro_phenomenon"], positive["critical_word"], candidate["critical_word"]))
    pair_words = sorted({w for _, a, b in critical_pairs for w in (a, b)})

    report: dict[str, Any] = {"purpose": PURPOSE, "max_seq_length": MAX_SEQ_LENGTH,
                              "n_words": len(words), "n_morph_hard_pairs": len(critical_pairs), "arms": {}}
    base_lengths = None
    for arm in ARMS:
        segments = arm_cache(arm, caches, words + pair_words) or {}

        def pieces_of(word: str) -> list[str]:
            parts = lookup(segments, word) if segments else None
            return tokenizer.tokenize(MARKER.join(parts)) if parts and len(parts) > 1 else tokenizer.tokenize(word)

        arm_pieces = {w: pieces_of(w) for w in words}
        changed = [w for w in words if arm_pieces[w] != base_pieces[w]]
        marked = mark_items(items, segments or None)
        texts = [("query: " + it["query"]) for it in marked] + \
                [("passage: " + c["text"]) for it in marked for c in it["candidates"]]
        lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
        if arm == "base":
            base_lengths = lengths
        seg_cuts_default = [len({sum(map(len, parts[:i])) for i in range(1, len(parts))}
                                & cut_positions(base_pieces[w]))
                            / max(1, len(parts) - 1) for w, parts in segments.items() if w in base_pieces and len(parts) > 1]
        per_macro: dict[str, list[tuple[int, int]]] = defaultdict(list)
        all_pairs = []
        for macro, pos_word, hard_word in critical_pairs:
            a, b = pieces_of(pos_word), pieces_of(hard_word)
            shared = common_prefix_len(a, b)
            row = (int(shared >= 1 and len(a) + len(b) - 2 * shared > 0 and shared < min(len(a), len(b)) + 1),
                   len(a) + len(b) - 2 * shared)
            per_macro[macro].append(row)
            all_pairs.append(row)
        report["arms"][arm] = {
            "words_with_changed_tokenization": round(len(changed) / len(words), 4),
            "tokens_per_text_mean": round(mean(lengths), 2),
            "length_vs_base": round(mean(lengths) / mean(base_lengths), 4),
            "texts_over_max_seq_length": round(sum(n > MAX_SEQ_LENGTH for n in lengths) / len(lengths), 5),
            "max_text_tokens": max(lengths),
            "tool_boundaries_already_in_default_tokenization": round(mean(seg_cuts_default), 4) if seg_cuts_default else None,
            "morph_hard_pairs": {
                "root_piece_shared_rate": round(mean(r[0] for r in all_pairs), 4),
                "differing_pieces_mean": round(mean(r[1] for r in all_pairs), 3),
                "by_macro": {m: {"root_piece_shared_rate": round(mean(r[0] for r in rows), 4),
                                 "differing_pieces_mean": round(mean(r[1] for r in rows), 3), "n": len(rows)}
                             for m, rows in sorted(per_macro.items())},
            },
        }
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Tokenizer probe: tokenizasyon denetimi", "",
             f"purpose: `{report['purpose']}`; {report['n_words']} benzersiz kelime, "
             f"{report['n_morph_hard_pairs']} (positive, morph-hard) kritik kelime çifti.", "",
             "| kol | değişen kelime | token/metin | uzunluk (base=1) | >256 token | araç sınırı zaten varsayılan | çiftte ortak kök parçası | çiftte farklı parça |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, r in report["arms"].items():
        pairs = r["morph_hard_pairs"]
        lines.append(f"| {arm} | {r['words_with_changed_tokenization']:.1%} | {r['tokens_per_text_mean']} | "
                     f"{r['length_vs_base']} | {r['texts_over_max_seq_length']:.2%} | "
                     f"{'-' if r['tool_boundaries_already_in_default_tokenization'] is None else format(r['tool_boundaries_already_in_default_tokenization'], '.1%')} | "
                     f"{pairs['root_piece_shared_rate']:.1%} | {pairs['differing_pieces_mean']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    report = audit(args.dir)
    (args.dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.dir / "audit_report.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
