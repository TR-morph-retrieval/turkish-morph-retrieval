"""Aggregate out-of-fold results into the per-phenomenon report and the pre-registered verdicts.

    python -m experiments.tokenizer_probe.report [--runs runs] [--out report_out]

Everything is tagged ``tokenizer_probe_only``: the sealed 600 were used for training here.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from test.evaluation import holm_adjust, mcnemar

from .data import ARTIFACTS, PURPOSE, RUNS, load_items

METRICS = ("recall@1", "mrr@10", "pairwise_morph_hard_accuracy", "hardest_hard_margin")
EFFECT_BAND = 0.03   # pre-registered: practical-equivalence band and minimum effect on MRR@10
N_BOOT = 10000


def _num(value) -> float:
    return float("nan") if value is None else float(value)


def load_runs(runs: Path) -> tuple[dict[str, dict[str, dict]], dict[str, dict[int, dict[str, dict]]]]:
    """zero_shot[arm][family] -> row; lora[arm][seed][family] -> row (out-of-fold test rows)."""
    zero: dict[str, dict[str, dict]] = {}
    for path in sorted((runs / "zero_shot").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        zero[payload["arm"]] = {r["query_id"]: r for r in payload["per_query"]}
    lora: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(runs.glob("*/fold*_seed*.json")):
        if path.parent.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["per_query"]:
            lora[payload["arm"]][payload["seed"]][row["query_id"]] = {**row, "_fold": payload["fold"]}
    return zero, {a: dict(s) for a, s in lora.items()}


def per_family(rows_by_seed: dict[int, dict[str, dict]], metric: str) -> dict[str, float]:
    """Seed-averaged metric per family."""
    families = set.intersection(*(set(r) for r in rows_by_seed.values()))
    return {f: float(np.nanmean([_num(rows[f].get(metric)) for rows in rows_by_seed.values()])) for f in families}


def cluster_bootstrap(diff: np.ndarray, groups: list[str], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean paired difference, resampling lemma groups (ratio estimator)."""
    keys = sorted(set(groups))
    index = {k: i for i, k in enumerate(keys)}
    sums, counts = np.zeros(len(keys)), np.zeros(len(keys))
    for d, g in zip(diff, groups):
        sums[index[g]] += d
        counts[index[g]] += 1
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def verdict(delta: float, low: float, high: float, p_holm: float, folds_positive: int, n_folds: int) -> str:
    if high < 0:
        return "zararlı"
    if delta >= EFFECT_BAND and low > 0 and p_holm < 0.05 and folds_positive >= n_folds - 1:
        return "etkili (kontrol kolu Aşama C'de doğrulanmalı)"
    if low >= -EFFECT_BAND and high <= EFFECT_BAND:
        return "pratikte fark yok"
    return "belirsiz (güç yetersiz)"


def binarize(rows_by_seed: dict[int, dict[str, dict]]) -> dict[str, int]:
    values = per_family(rows_by_seed, "recall@1")
    return {f: int(v >= 0.5) for f, v in values.items()}


def compare(lora, items_by_id, manifest) -> dict[str, dict[str, Any]]:
    arms = [a for a in lora if a != "base"]
    if "base" not in lora:
        return {}
    base_mrr = per_family(lora["base"], "mrr@10")
    base_bin = binarize(lora["base"])
    out: dict[str, dict[str, Any]] = {}
    pvals = {}
    for arm in arms:
        mrr = per_family(lora[arm], "mrr@10")
        ids = sorted(set(mrr) & set(base_mrr))
        diff = np.array([mrr[i] - base_mrr[i] for i in ids])
        groups = [items_by_id[i]["critical_lemma"] for i in ids]
        low, high = cluster_bootstrap(diff, groups)
        arm_bin = binarize(lora[arm])
        test = mcnemar([arm_bin[i] for i in ids], [base_bin[i] for i in ids])
        fold_of = manifest["fold_of"]
        per_fold = [float(np.mean([d for d, i in zip(diff, ids) if fold_of[i] == k])) for k in range(manifest["n_folds"])
                    if any(fold_of[i] == k for i in ids)]
        pvals[arm] = test["exact_p"]
        out[arm] = {"n": len(ids), "delta_mrr@10": float(diff.mean()), "ci95": [low, high],
                    "delta_recall@1": float(np.mean([arm_bin[i] - base_bin[i] for i in ids])), "mcnemar": test,
                    "per_fold_delta_mrr": per_fold, "folds_positive": sum(d > 0 for d in per_fold),
                    "diff_sd": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")}
    holm = holm_adjust(pvals) if pvals else {}
    for arm, info in out.items():
        info["p_holm"] = holm[arm]
        info["verdict"] = verdict(info["delta_mrr@10"], *info["ci95"], holm[arm], info["folds_positive"], len(info["per_fold_delta_mrr"]))
    return out


def phenomenon_rows(items, zero, lora) -> list[dict[str, Any]]:
    meta = {i["family_id"]: i for i in items}
    arms = list(lora)
    lora_r1 = {a: per_family(lora[a], "recall@1") for a in arms}
    lora_mrr = {a: per_family(lora[a], "mrr@10") for a in arms}
    rows = []
    for feature in sorted({i["target_feature"] for i in items}, key=lambda f: (next(i["macro_phenomenon"] for i in items if i["target_feature"] == f), f)):
        ids = [i["family_id"] for i in items if i["target_feature"] == feature]
        first = meta[ids[0]]
        row: dict[str, Any] = {"target_feature": feature, "label": first["target_feature_label"],
                               "macro_phenomenon": first["macro_phenomenon"], "objective": first["objective"], "n": len(ids)}
        for arm, rows_ in zero.items():
            row[f"zs_{arm}_recall@1"] = round(float(np.mean([rows_[i]["recall@1"] for i in ids])), 3)
        for arm in arms:
            have = [i for i in ids if i in lora_r1[arm]]
            row[f"lora_{arm}_recall@1"] = round(float(np.mean([lora_r1[arm][i] for i in have])), 3) if have else ""
            if arm != "base" and "base" in lora_mrr:
                d = [lora_mrr[arm][i] - lora_mrr["base"][i] for i in have if i in lora_mrr["base"]]
                row[f"{arm}_vs_base_W/L/T"] = f"{sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}/{sum(abs(x) <= 1e-9 for x in d)}"
        rows.append(row)
    return rows


def group_table(items, lora, field: str) -> dict[str, dict[str, dict[str, float]]]:
    meta = {i["family_id"]: i for i in items}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm in lora:
        r1, mrr = per_family(lora[arm], "recall@1"), per_family(lora[arm], "mrr@10")
        groups = defaultdict(list)
        for f in r1:
            groups[str(meta[f][field])].append(f)
        out[arm] = {g: {"n": len(fs), "recall@1": float(np.mean([r1[f] for f in fs])), "mrr@10": float(np.mean([mrr[f] for f in fs]))}
                    for g, fs in sorted(groups.items())}
    return out


def overall(zero, lora) -> list[dict[str, Any]]:
    rows = []
    for arm, rows_ in zero.items():
        rows.append({"arm": arm, "stage": "zero-shot (600)", "n": len(rows_),
                     **{m: round(float(np.nanmean([_num(r.get(m)) for r in rows_.values()])), 4) for m in METRICS}})
    for arm, by_seed in lora.items():
        values = {m: per_family(by_seed, m) for m in METRICS}
        rows.append({"arm": arm, "stage": f"LoRA out-of-fold ({len(by_seed)} seed)", "n": len(values["recall@1"]),
                     **{m: round(float(np.nanmean(list(v.values()))), 4) for m, v in values.items()}})
    return rows


def write_markdown(path: Path, over, comparisons, phen, macro, objective) -> None:
    lines = ["# Tokenizer probe raporu (mE5-large + morfem sınırları)", "",
             f"purpose: `{PURPOSE}`. 600 sealed family eğitim/seçim için kullanıldı; bunlar sealed-test sonucu değildir.", "",
             "## Genel (out-of-fold test, tüm family'ler)", "",
             "| kol | aşama | n | recall@1 | MRR@10 | morph-hard pairwise acc | hardest-hard margin |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['arm']} | {r['stage']} | {r['n']} | {r['recall@1']} | {r['mrr@10']} | {r['pairwise_morph_hard_accuracy']} | {r['hardest_hard_margin']} |" for r in over]
    lines += ["", "## Karar (ön-kayıtlı kural, base'e karşı, ΔMRR@10)", "",
              "| kol | n | ΔMRR@10 | %95 CI (lemma-kümeli) | Δrecall@1 | McNemar p (Holm) | pozitif fold | karar |", "|---|---|---|---|---|---|---|---|"]
    for arm, c in comparisons.items():
        lines.append(f"| {arm} | {c['n']} | {c['delta_mrr@10']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['delta_recall@1']:+.4f} | "
                     f"{c['p_holm']:.3f} | {c['folds_positive']}/{len(c['per_fold_delta_mrr'])} | {c['verdict']} |")
    if comparisons:
        sd = np.nanmean([c["diff_sd"] for c in comparisons.values()])
        lines += ["", f"Ölçülen aile-başı ΔMRR SD ≈ {sd:.3f}; n=600 için minimum saptanabilir etki ≈ {2.8 * sd / np.sqrt(600):.3f} (plan varsayımı 0,035)."]
    for title, table in (("macro_phenomenon", macro), ("objective", objective)):
        arms = list(table)
        if not arms:
            continue
        lines += ["", f"## {title} kırılımı (LoRA out-of-fold recall@1 / MRR@10)", "", "| grup | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for g in table[arms[0]]:
            lines.append(f"| {g} | {table[arms[0]][g]['n']} | " + " | ".join(f"{table[a][g]['recall@1']:.3f} / {table[a][g]['mrr@10']:.3f}" for a in arms) + " |")
    lines += ["", "## Fenomen düzeyi (n küçük, betimleyici; W/L/T = kol, base'e göre MRR kazanç/kayıp/beraberlik)", ""]
    if phen:
        cols = list(phen[0])
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in phen]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def heatmap(path: Path, phen) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in phen[0] if c.startswith("lora_") and c.endswith("recall@1")]
    if not cols:
        return
    data = np.array([[float(r[c]) if r[c] != "" else np.nan for c in cols] for r in phen])
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.22 * len(phen) + 1.5))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cols)), [c.replace("lora_", "").replace("_recall@1", "") for c in cols])
    ax.set_yticks(range(len(phen)), [f"{r['target_feature']} (n={r['n']})" for r in phen], fontsize=6)
    fig.colorbar(im, ax=ax, label="recall@1 (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.runs / "report"
    out.mkdir(parents=True, exist_ok=True)
    items = load_items()
    manifest = json.loads((args.artifacts / "split_manifest.json").read_text(encoding="utf-8"))
    zero, lora = load_runs(args.runs)
    by_id = {i["family_id"]: i for i in items}
    comparisons = compare(lora, by_id, manifest)
    phen = phenomenon_rows(items, zero, lora) if (zero or lora) else []
    over = overall(zero, lora)
    macro, objective = group_table(items, lora, "macro_phenomenon"), group_table(items, lora, "objective")
    write_markdown(out / "report.md", over, comparisons, phen, macro, objective)
    if phen:
        with (out / "per_phenomenon.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(phen[0]))
            w.writeheader()
            w.writerows(phen)
        heatmap(out / "per_phenomenon_heatmap.png", phen)
    (out / "comparisons.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=1), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8")[:3500])
    print("yazıldı:", out)


if __name__ == "__main__":
    main()
