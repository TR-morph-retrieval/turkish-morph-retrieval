"""Reusable closed/full-corpus metrics, artifact baselines, ablations and paired statistics."""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .taxonomy import MORPH_HARD_SUBTYPES, SEMANTIC_HARD_SUBTYPES
from .validators import _bm25_scores, char_ngrams, jaccard, tokens

EVALUATION_API_VERSION = "3.2"
CONTROLLED_RECALL_KS = (1, 3)
FULL_CORPUS_RECALL_KS = (1, 3, 10, 50)


def load_items(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) and "items" in data else data


def closed_qrels(items: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        item["family_id"]: {
            str(candidate_id): float(score)
            for candidate_id, score in item.get("qrels", {item["gold_id"]: 1}).items()
        }
        for item in items
    }


def load_qrels(path: str | Path) -> dict[str, dict[str, float]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {
            query_id: {
                row["corpus_id"]: float(row.get("relevance", row.get("score", 0)))
                for row in rows if row["query_id"] == query_id
            }
            for query_id in sorted({row["query_id"] for row in rows})
        }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        out[str(row["query-id"])][str(row["corpus-id"])] = float(row["score"])
    return dict(out)


def validate_binary_qrels(qrels: dict[str, dict[str, float]], require_nonrelevant: bool = True) -> None:
    values = [score for judgments in qrels.values() for score in judgments.values()]
    invalid = sorted({score for score in values if score not in {0, 0.0, 1, 1.0}})
    if invalid:
        raise ValueError(f"Binary qrels yalnız 0/1 içerebilir; bulunan: {invalid}")
    if not all(any(score == 1 for score in judgments.values()) for judgments in qrels.values()):
        raise ValueError("Her query için en az bir relevant=1 judgment gerekli")
    if require_nonrelevant and not any(score == 0 for score in values):
        raise ValueError("Binary qrels içinde en az bir açık negative=0 etiketi gerekli")


def evaluate_run(
    qrels: dict[str, dict[str, float]],
    run: dict[str, list[str]],
    recall_ks: tuple[int, ...] = CONTROLLED_RECALL_KS,
    unjudged_policy: str = "nonrelevant",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if unjudged_policy not in {"nonrelevant", "condensed"}:
        raise ValueError("unjudged_policy nonrelevant veya condensed olmalı")
    per_query = []
    for query_id, relevant in qrels.items():
        original_ranked = run.get(query_id, [])
        ranked = (
            [doc_id for doc_id in original_ranked if doc_id in relevant]
            if unjudged_policy == "condensed" else original_ranked
        )
        positives = {doc_id for doc_id, score in relevant.items() if score > 0}
        if not positives:
            continue
        gains = [relevant.get(doc_id, 0.0) for doc_id in ranked[:10]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_gains = sorted(relevant.values(), reverse=True)[:10]
        ideal = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
        first_rank = next((index + 1 for index, doc_id in enumerate(ranked) if doc_id in positives), None)
        row = {
            "query_id": query_id,
            "rank": first_rank,
            "mrr@10": 1.0 / first_rank if first_rank and first_rank <= 10 else 0.0,
            "ndcg@10": dcg / ideal if ideal else 0.0,
        }
        for k in recall_ks:
            row[f"recall@{k}"] = len(positives & set(ranked[:k])) / len(positives)
            if unjudged_policy == "condensed":
                row[f"judged@{k}"] = (
                    sum(doc_id in relevant for doc_id in original_ranked[:k]) / max(1, min(k, len(original_ranked)))
                )
        per_query.append(row)
    metric_names = [key for key in per_query[0] if key not in {"query_id", "rank"}] if per_query else []
    summary = {name: sum(row[name] for row in per_query) / len(per_query) for name in metric_names}
    valid_ranks = [row["rank"] for row in per_query if row["rank"] is not None]
    if valid_ranks:
        ordered_ranks = sorted(valid_ranks)
        midpoint = len(ordered_ranks) // 2
        mean_rank = sum(ordered_ranks) / len(ordered_ranks)
        median_rank = (
            float(ordered_ranks[midpoint]) if len(ordered_ranks) % 2
            else (ordered_ranks[midpoint - 1] + ordered_ranks[midpoint]) / 2
        )
        if len(valid_ranks) == len(per_query):
            summary["mean_rank"] = mean_rank
            summary["median_rank"] = median_rank
        else:
            summary["mean_rank_retrieved"] = mean_rank
            summary["median_rank_retrieved"] = median_rank
            summary["retrieved_ratio"] = len(valid_ranks) / max(1, len(per_query))
    return summary, per_query


def score_encoder(
    model,
    items: list[dict[str, Any]],
    query_prefix: str = "",
    document_prefix: str = "",
    full_corpus: bool = False,
    qrels: dict[str, dict[str, float]] | None = None,
    batch_size: int = 32,
    include_full_run: bool = False,
    full_run_depth: int | None = None,
    unjudged_policy: str = "nonrelevant",
) -> dict[str, Any]:
    """Encode once, then rank own candidates and optionally the shared corpus too.

    ``full_run_depth`` limits only the stored shared-corpus ranking.  Top 100 is sufficient for
    this benchmark's official metrics and avoids serialising 2.75 million document IDs per model
    in the 500-query/5,500-document final evaluation.
    """
    import numpy as np

    queries = [query_prefix + item["query"] for item in items]
    documents = [(candidate["id"], candidate["text"]) for item in items for candidate in item["candidates"]]
    query_embeddings = model.encode(queries, normalize_embeddings=True, batch_size=batch_size)
    document_embeddings = model.encode(
        [document_prefix + text for _, text in documents], normalize_embeddings=True, batch_size=batch_size
    )
    doc_ids = [doc_id for doc_id, _ in documents]
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    run = {}
    full_run = {} if include_full_run or full_corpus else None
    own_scores_by_query: dict[str, dict[str, float]] = {}
    for query_index, item in enumerate(items):
        candidate_ids = doc_ids if full_corpus else [candidate["id"] for candidate in item["candidates"]]
        indices = [doc_index[doc_id] for doc_id in candidate_ids]
        scores = document_embeddings[indices] @ query_embeddings[query_index]
        order = np.argsort(-scores)
        ranking = [candidate_ids[index] for index in order]
        run[item["family_id"]] = (
            ranking[:full_run_depth]
            if full_corpus and full_run_depth is not None else ranking
        )
        if full_run is not None:
            if full_corpus:
                ranking = run[item["family_id"]]
                full_run[item["family_id"]] = (
                    ranking[:full_run_depth] if full_run_depth is not None else ranking
                )
            else:
                all_scores = document_embeddings @ query_embeddings[query_index]
                all_order = np.argsort(-all_scores)
                if full_run_depth is not None:
                    all_order = all_order[:full_run_depth]
                full_run[item["family_id"]] = [doc_ids[index] for index in all_order]
        own_scores_by_query[item["family_id"]] = {
            candidate["id"]: float(document_embeddings[doc_index[candidate["id"]]] @ query_embeddings[query_index])
            for candidate in item["candidates"]
        }
    resolved_qrels = qrels if qrels is not None else closed_qrels(items)
    recall_ks = FULL_CORPUS_RECALL_KS if full_corpus else CONTROLLED_RECALL_KS
    summary, per_query = evaluate_run(
        resolved_qrels, run, recall_ks=recall_ks, unjudged_policy=unjudged_policy
    )
    row_by_query = {row["query_id"]: row for row in per_query}
    for item in items:
        row = row_by_query.get(item["family_id"])
        if row is None or "gold_id" not in item:
            continue
        family_scores = own_scores_by_query[item["family_id"]]
        gold_id = item["gold_id"]
        gold_score = family_scores[gold_id]
        hard_candidates = [candidate for candidate in item["candidates"] if candidate.get("role") == "hard_negative"]
        easy_candidates = [candidate for candidate in item["candidates"] if candidate.get("role") == "easy_negative"]
        minimal_candidates = [
            candidate for candidate in hard_candidates if candidate.get("subtype") == "minimal_morph_negative"
        ]
        morph_hards = [
            candidate for candidate in hard_candidates
            if candidate.get("subtype") in MORPH_HARD_SUBTYPES
        ]
        semantic_hards = [
            candidate for candidate in hard_candidates
            if candidate.get("subtype") in SEMANTIC_HARD_SUBTYPES
        ]
        hard_ranking = sorted(
            [gold_id] + [candidate["id"] for candidate in hard_candidates],
            key=lambda doc_id: (-family_scores[doc_id], doc_id),
        )
        hard_rank = hard_ranking.index(gold_id) + 1

        def pairwise_win(candidate: dict[str, Any]) -> float:
            candidate_score = family_scores[candidate["id"]]
            return 1.0 if gold_score > candidate_score else 0.5 if gold_score == candidate_score else 0.0

        row.update({
            "hard_rank": hard_rank,
            "hard_only_mrr@10": 1.0 / hard_rank if hard_rank <= 10 else 0.0,
            "hard_only_ndcg@10": 1.0 / math.log2(hard_rank + 1) if hard_rank <= 10 else 0.0,
            "pairwise_hard_accuracy": (
                sum(pairwise_win(candidate) for candidate in hard_candidates) / len(hard_candidates)
                if hard_candidates else 0.0
            ),
            "pairwise_easy_accuracy": (
                sum(pairwise_win(candidate) for candidate in easy_candidates) / len(easy_candidates)
                if easy_candidates else 0.0
            ),
            "pairwise_morph_hard_accuracy": (
                sum(pairwise_win(candidate) for candidate in morph_hards) / len(morph_hards)
                if morph_hards else None
            ),
            "pairwise_semantic_hard_accuracy": (
                sum(pairwise_win(candidate) for candidate in semantic_hards) / len(semantic_hards)
                if semantic_hards else None
            ),
            "morph_hard_family_consistency": (
                float(all(pairwise_win(candidate) == 1.0 for candidate in morph_hards))
                if morph_hards else None
            ),
            "semantic_hard_family_consistency": (
                float(all(pairwise_win(candidate) == 1.0 for candidate in semantic_hards))
                if semantic_hards else None
            ),
            "all_hard_family_consistency": float(
                bool(hard_candidates) and all(pairwise_win(candidate) == 1.0 for candidate in hard_candidates)
            ),
            "pairwise_all_accuracy": (
                sum(pairwise_win(candidate) for candidate in hard_candidates + easy_candidates)
                / len(hard_candidates + easy_candidates)
                if hard_candidates or easy_candidates else 0.0
            ),
            "contrast_consistency": float(
                bool(minimal_candidates) and all(pairwise_win(candidate) == 1.0 for candidate in minimal_candidates)
            ),
            "positive_score": gold_score,
            "minimal_margin": (
                gold_score - max(family_scores[candidate["id"]] for candidate in minimal_candidates)
                if minimal_candidates else None
            ),
            "hardest_hard_margin": (
                gold_score - max(family_scores[candidate["id"]] for candidate in hard_candidates)
                if hard_candidates else None
            ),
            "hardest_negative_margin": (
                gold_score - max(
                    family_scores[candidate["id"]] for candidate in hard_candidates + easy_candidates
                ) if hard_candidates or easy_candidates else None
            ),
        })
    for metric in (
        "hard_only_mrr@10", "hard_only_ndcg@10",
        "pairwise_hard_accuracy", "pairwise_easy_accuracy", "pairwise_all_accuracy",
        "pairwise_morph_hard_accuracy", "pairwise_semantic_hard_accuracy",
        "morph_hard_family_consistency", "semantic_hard_family_consistency",
        "all_hard_family_consistency",
        "contrast_consistency", "minimal_margin", "hardest_hard_margin", "hardest_negative_margin",
    ):
        values = [row[metric] for row in per_query if metric in row]
        values = [value for value in values if value is not None]
        if values:
            summary[metric] = sum(values) / len(values)
    hard_ranks = [row["hard_rank"] for row in per_query if "hard_rank" in row]
    if hard_ranks:
        summary["mean_hard_rank"] = sum(hard_ranks) / len(hard_ranks)
    result = {"summary": summary, "per_query": per_query, "run": run, "scores": own_scores_by_query}
    if full_run is not None:
        result["full_run"] = full_run
    return result


def _word_overlap(query: str, document: str) -> float:
    query_terms = set(tokens(query))
    return len(query_terms & set(tokens(document))) / max(1, len(query_terms))


def _prefix5_overlap(query: str, document: str) -> float:
    """Cheap suffix-reduction control; intentionally not presented as a linguistic lemmatiser."""
    query_prefixes = {token[:5] for token in tokens(query)}
    document_prefixes = {token[:5] for token in tokens(document)}
    return len(query_prefixes & document_prefixes) / max(1, len(query_prefixes))


def _artifact_scores(item: dict[str, Any], learned_position: int | None = None) -> dict[str, list[float]]:
    candidates = item["candidates"]
    texts = [candidate["text"] for candidate in candidates]
    scorers: dict[str, list[float]] = {
        "longest_candidate": [float(len(text)) for text in texts],
        "most_tokens": [float(len(tokens(text))) for text in texts],
        "character_3gram": [jaccard(char_ngrams(item["query"]), char_ngrams(text)) for text in texts],
        "word_overlap": [_word_overlap(item["query"], text) for text in texts],
        "prefix5_overlap": [_prefix5_overlap(item["query"], text) for text in texts],
        "bm25": _bm25_scores(item["query"], texts),
    }
    if learned_position is not None:
        scorers["position_only"] = [
            float(index == learned_position) for index in range(len(candidates))
        ]
    return scorers


def artifact_baseline_runs(
    items: list[dict[str, Any]], learned_position: int | None = None
) -> dict[str, dict[str, list[str]]]:
    """Deterministic rankings for inspection; paper summaries use tie-aware expectations."""
    runs = defaultdict(dict)
    for item in items:
        candidates = item["candidates"]
        ids = [candidate["id"] for candidate in candidates]
        for name, scores in _artifact_scores(item, learned_position).items():
            order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
            runs[name][item["family_id"]] = [ids[index] for index in order]
    return dict(runs)


def _tie_aware_metrics(scores: list[float], gold_index: int) -> dict[str, float]:
    """Expected single-gold metrics under uniform ordering inside an exact score tie."""
    gold_score = scores[gold_index]
    higher = sum(score > gold_score for score in scores)
    tied = sum(score == gold_score for score in scores)
    possible_ranks = range(higher + 1, higher + tied + 1)
    row = {
        "mrr@10": sum(1.0 / rank if rank <= 10 else 0.0 for rank in possible_ranks) / tied,
        "ndcg@10": sum(
            1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in possible_ranks
        ) / tied,
        "mean_rank": higher + (tied + 1) / 2,
    }
    for k in CONTROLLED_RECALL_KS:
        row[f"recall@{k}"] = max(0, min(tied, k - higher)) / tied
    return row


def artifact_baseline_summaries(
    items: list[dict[str, Any]], learned_position: int | None = None
) -> dict[str, dict[str, float]]:
    """Tie-aware cheap-baseline summaries without candidate-ID tie artefacts."""
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in items:
        gold_index = next(
            index for index, candidate in enumerate(item["candidates"])
            if candidate["id"] == item["gold_id"]
        )
        for name, scores in _artifact_scores(item, learned_position).items():
            rows[name].append(_tie_aware_metrics(scores, gold_index))
    return {
        name: {
            metric: sum(row[metric] for row in metric_rows) / len(metric_rows)
            for metric in metric_rows[0]
        }
        for name, metric_rows in rows.items()
    }


def learn_gold_position(dev_items: list[dict[str, Any]]) -> int:
    counts = Counter(
        next(index for index, candidate in enumerate(item["candidates"]) if candidate["id"] == item["gold_id"])
        for item in dev_items
    )
    return counts.most_common(1)[0][0]


def evaluate_artifacts(dev_items: list[dict], test_items: list[dict]) -> dict[str, dict[str, float]]:
    position = learn_gold_position(dev_items)
    return artifact_baseline_summaries(test_items, position)


def candidate_only_classifier(dev_items: list[dict], test_items: list[dict]) -> dict[str, Any]:
    """Query-blind char-TFIDF classifier, trained only on development candidates."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_texts = [candidate["text"] for item in dev_items for candidate in item["candidates"]]
    train_labels = [int(candidate["id"] == item["gold_id"]) for item in dev_items for candidate in item["candidates"]]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000)
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    classifier.fit(vectorizer.fit_transform(train_texts), train_labels)
    run = {}
    for item in test_items:
        texts = [candidate["text"] for candidate in item["candidates"]]
        scores = classifier.predict_proba(vectorizer.transform(texts))[:, 1]
        order = sorted(range(len(texts)), key=lambda index: (-scores[index], item["candidates"][index]["id"]))
        run[item["family_id"]] = [item["candidates"][index]["id"] for index in order]
    summary, per_query = evaluate_run(
        closed_qrels(test_items), run, recall_ks=CONTROLLED_RECALL_KS
    )
    return {"summary": summary, "per_query": per_query, "run": run}


_TR_CHAR_CLASS = r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]"


def _remove_once(text: str, word: str) -> str:
    import re
    if not word or not word.strip():
        return text
    cleaned_word = word.strip()
    escaped = re_escape_word(cleaned_word)
    pattern = rf"(?<!{_TR_CHAR_CLASS}){escaped}(?!{_TR_CHAR_CLASS})"
    replaced, n = re.subn(pattern, "[MASK]", text, count=1, flags=re.IGNORECASE)
    if n == 0:
        pattern = re_escape_word(cleaned_word)
        replaced = re.sub(pattern, "[MASK]", text, count=1, flags=re.IGNORECASE)
    return " ".join(replaced.split())


def re_escape_word(word: str) -> str:
    import re
    return re.escape(word.strip())


def ablate_items(items: list[dict], mode: str) -> list[dict]:
    """Remove the critical word or truncate tokens to five-character prefixes.

    Prefix truncation is only a cheap suffix-reduction control.  It is not Turkish morphological
    analysis and must not be reported as a true roots-only or lemma-only experiment.
    """
    if mode not in {"critical_deleted", "prefix5"}:
        raise ValueError("mode critical_deleted veya prefix5 olmalı")
    out = copy.deepcopy(items)
    for item in out:
        if mode == "prefix5":
            item["query"] = " ".join(token[:5] for token in tokens(item["query"]))
            for candidate in item["candidates"]:
                candidate["text"] = " ".join(token[:5] for token in tokens(candidate["text"]))
        else:
            item["query"] = _remove_once(item["query"], item.get("critical_word_query", ""))
            for candidate in item["candidates"]:
                candidate["text"] = _remove_once(candidate["text"], candidate.get("critical_word", ""))
    return out


def bootstrap_ci(values, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    import numpy as np
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(draws, [2.5, 97.5]))


def paired_bootstrap(left, right, n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    import numpy as np
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired vectors aynı shape'te olmalı")
    differences = left - right
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(n_boot, len(differences)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"mean_difference": float(differences.mean()), "ci_low": float(low), "ci_high": float(high)}


def approximate_randomization(left, right, n_iter: int = 10000, seed: int = 0) -> float:
    import numpy as np

    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.size == 0:
        raise ValueError("paired vectors boş olamaz")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    # Chunking keeps memory bounded for the 500-query final test while avoiding slow Python loops.
    for start in range(0, n_iter, 1000):
        size = min(1000, n_iter - start)
        signs = rng.choice((-1.0, 1.0), size=(size, differences.size))
        randomized = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed))
    return (extreme + 1) / (n_iter + 1)


def mcnemar(left_binary, right_binary) -> dict[str, float]:
    left_binary, right_binary = list(left_binary), list(right_binary)
    b = sum(int(left) == 1 and int(right) == 0 for left, right in zip(left_binary, right_binary))
    c = sum(int(left) == 0 and int(right) == 1 for left, right in zip(left_binary, right_binary))
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        p_value = min(1.0, 2 * tail)
    return {
        "left_only_correct": b,
        "right_only_correct": c,
        "exact_p": p_value,
        "matched_odds_ratio": (b + 0.5) / (c + 0.5),
        "risk_difference": (b - c) / max(1, len(left_binary)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def slice_summary(per_query: list[dict], items: list[dict], metric: str = "recall@1") -> dict[str, Any]:
    metadata = {item["family_id"]: item for item in items}
    out = {}
    for field in (
        "query_sentence_count", "passage_sentence_count", "critical_sentence_position",
        "layer", "objective", "generalization_bucket", "macro_phenomenon", "target_feature",
        "family_mode", "query_expression", "query_gold_lexical_band",
        "strict_minimal_pair", "generator_id",
    ):
        groups = defaultdict(list)
        for row in per_query:
            groups[str(metadata[row["query_id"]].get(field, "unknown"))].append(float(row[metric]))
        out[field] = {group: {"n": len(values), "mean": sum(values) / len(values)} for group, values in sorted(groups.items())}
    return out
