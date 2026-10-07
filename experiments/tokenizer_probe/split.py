"""Deterministic 5-fold rotating train/val/test split over the sealed 600.

Stratified by ``target_feature`` (76 phenomena), grouped by ``critical_lemma`` so a lemma never
straddles train and test. Fold k: test = fold k, val = fold (k+1) mod n, train = the rest. Every
family is a test family exactly once, so out-of-fold results cover all 600 families.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold

from .data import ARTIFACTS, PURPOSE, SEALED_PATH, SEALED_SHA256, file_sha256, load_sealed

N_FOLDS = 5
SEED = 42
MANIFEST_NAME = "split_manifest.json"
COVERAGE_NAME = "phenomenon_coverage.csv"


def build_split(items: list[dict[str, Any]], n_folds: int = N_FOLDS, seed: int = SEED,
                data_repairs: dict[str, Any] | None = None) -> dict[str, Any]:
    family_ids = [item["family_id"] for item in items]
    labels = [item["target_feature"] for item in items]
    groups = [item["critical_lemma"] for item in items]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of: dict[str, int] = {}
    with warnings.catch_warnings():
        # 20 phenomena have fewer than 5 families; sklearn warns, the split is still valid.
        warnings.simplefilter("ignore", UserWarning)
        for fold, (_, test_index) in enumerate(splitter.split(family_ids, labels, groups)):
            for index in test_index:
                fold_of[family_ids[index]] = fold
    folds = []
    for fold in range(n_folds):
        val_fold = (fold + 1) % n_folds
        folds.append({
            "fold": fold,
            "test": sorted(f for f, k in fold_of.items() if k == fold),
            "val": sorted(f for f, k in fold_of.items() if k == val_fold),
            "train": sorted(f for f, k in fold_of.items() if k not in (fold, val_fold)),
        })
    return {
        "purpose": PURPOSE,
        "source": str(SEALED_PATH.name),
        "source_sha256": SEALED_SHA256,
        "data_repairs": data_repairs or {},
        "n_folds": n_folds,
        "seed": seed,
        "stratify_by": "target_feature",
        "group_by": "critical_lemma",
        "fold_of": fold_of,
        "folds": folds,
    }


def validate_split(manifest: dict[str, Any], items: list[dict[str, Any]]) -> None:
    """Raise if the split breaks an invariant the experiment relies on."""
    ids = {item["family_id"] for item in items}
    lemma_of = {item["family_id"]: item["critical_lemma"] for item in items}
    if set(manifest["fold_of"]) != ids:
        raise ValueError("fold_of tüm family'leri tam bir kez içermeli")
    test_count = Counter()
    for fold in manifest["folds"]:
        parts = {name: set(fold[name]) for name in ("train", "val", "test")}
        if parts["train"] & parts["val"] or parts["train"] & parts["test"] or parts["val"] & parts["test"]:
            raise ValueError(f"fold {fold['fold']}: train/val/test kesişiyor")
        if parts["train"] | parts["val"] | parts["test"] != ids:
            raise ValueError(f"fold {fold['fold']}: bölmeler tüm family'leri kapsamıyor")
        for held_out in ("val", "test"):
            held_lemmas = {lemma_of[f] for f in parts[held_out]}
            if held_lemmas & {lemma_of[f] for f in parts["train"]}:
                raise ValueError(f"fold {fold['fold']}: lemma train ile {held_out} arasında bölünmüş")
        test_count.update(parts["test"])
    if set(test_count.values()) != {1} or set(test_count) != ids:
        raise ValueError("her family tam bir kez test olmalı")


def coverage_rows(manifest: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["family_id"]: item for item in items}
    meta: dict[str, dict[str, Any]] = {}
    for item in items:
        meta.setdefault(item["target_feature"], {
            "target_feature": item["target_feature"],
            "label": item["target_feature_label"],
            "macro_phenomenon": item["macro_phenomenon"],
            "objective": item["objective"],
        })
    counts: dict[str, dict[str, list[int]]] = {
        feature: {part: [0] * manifest["n_folds"] for part in ("train", "val", "test")} for feature in meta
    }
    for fold in manifest["folds"]:
        for part in ("train", "val", "test"):
            for family_id in fold[part]:
                counts[by_id[family_id]["target_feature"]][part][fold["fold"]] += 1
    rows = []
    for feature in sorted(meta, key=lambda f: (meta[f]["macro_phenomenon"], f)):
        row = dict(meta[feature])
        c = counts[feature]
        row["n_total"] = sum(c["test"])
        for fold in range(manifest["n_folds"]):
            row[f"test_f{fold}"] = c["test"][fold]
        row["train_min"] = min(c["train"])
        row["train_median"] = median(c["train"])
        row["val_min"] = min(c["val"])
        row["folds_with_test"] = sum(1 for n in c["test"] if n)
        rows.append(row)
    return rows


def write_artifacts(manifest: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / COVERAGE_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    sizes = [(len(f["train"]), len(f["val"]), len(f["test"])) for f in manifest["folds"]]
    low = [r["target_feature"] for r in rows if r["n_total"] < 5]
    no_train = [r["target_feature"] for r in rows if r["train_min"] == 0]
    lines = [f"fold (train/val/test): {sizes}",
             f"fenomen: {len(rows)}; n_total medyan {median(r['n_total'] for r in rows)}, "
             f"min {min(r['n_total'] for r in rows)}, max {max(r['n_total'] for r in rows)}",
             f"n_total < 5 olan fenomen: {len(low)}",
             f"bir fold'da train'de hiç örneği olmayan fenomen: {len(no_train)} {no_train}"]
    per_macro = defaultdict(lambda: [0] * manifest["n_folds"])
    for r in rows:
        for fold in range(manifest["n_folds"]):
            per_macro[r["macro_phenomenon"]][fold] += r[f"test_f{fold}"]
    lines += [f"  test sayıları {macro}: {counts}" for macro, counts in sorted(per_macro.items())]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    items, repairs = load_sealed()
    assert file_sha256(SEALED_PATH) == SEALED_SHA256
    manifest = build_split(items, data_repairs=repairs)
    validate_split(manifest, items)
    rows = coverage_rows(manifest, items)
    write_artifacts(manifest, rows, args.out)
    print(summarize(manifest, rows))
    print(f"yazıldı: {args.out / MANIFEST_NAME}, {args.out / COVERAGE_NAME}")


if __name__ == "__main__":
    main()
