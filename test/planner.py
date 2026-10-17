"""Prefix-stable deterministic coverage planner for raw test-family generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from .config import final_target
from .taxonomy import DOMAINS, FEATURES, MACROS, REGISTERS, TEMPLATES, hard_profile


def _seed(base: int, *parts: object) -> int:
    raw = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _weighted_value(
    distribution: dict[str, float], index: int, cycle_size: int, seed: int, salt: str
) -> str:
    """Pick from independently shuffled fixed-size cycles; prefixes never change with target size."""
    counts = {key: int(round(float(share) * cycle_size)) for key, share in distribution.items()}
    delta = cycle_size - sum(counts.values())
    order = sorted(distribution, key=lambda key: (-distribution[key], key))
    for offset in range(abs(delta)):
        key = order[offset % len(order)]
        counts[key] += 1 if delta > 0 else -1
    values = [key for key, count in counts.items() for _ in range(count)]
    cycle = index // cycle_size
    random.Random(_seed(seed, salt, cycle)).shuffle(values)
    return values[index % cycle_size]


def _weighted_occurrence_rank(
    distribution: dict[str, float], value: str, index: int, cycle_size: int, seed: int, salt: str
) -> int:
    """Zero-based rank of `value` among the prefix-stable weighted assignments."""
    per_cycle = int(round(float(distribution[value]) * cycle_size))
    cycle = index // cycle_size
    cycle_start = cycle * cycle_size
    earlier_in_cycle = sum(
        _weighted_value(distribution, prior, cycle_size, seed, salt) == value
        for prior in range(cycle_start, index)
    )
    return cycle * per_cycle + earlier_in_cycle


def _round_robin(values: tuple | list, index: int, seed: int, salt: str) -> Any:
    block = list(values)
    cycle = index // len(block)
    random.Random(_seed(seed, salt, cycle)).shuffle(block)
    return block[index % len(block)]


DEV_COMPOSITION_KEYS = {"NEG.AOR", "PLUPRF", "PST.PROG", "FUT.PST"}
NON_STRICT_FEATURE_KEYS = {
    "Q.PART.SCOPE", "MORPH.CONTEXT_AMBIG", "SUSP.AFFIX", "MWE.MORPH",
}
SEALED_DOMAIN_REGISTER_PAIRS = (
    ("health", "conversational"),
    ("finance", "news_report"),
    ("law_public_services", "everyday"),
    ("ecommerce", "formal_record"),
)


def _slot_id(slot: dict[str, Any], feature, cfg: dict[str, Any]) -> str:
    fingerprint = hashlib.sha256(
        f"{cfg['version']}|{slot['index']}|{feature.key}|{slot['generalization_bucket']}|"
        f"{slot['family_mode']}|{slot['query_expression']}|{slot['query_gold_lexical_band']}|"
        f"{slot['query_sentence_count']}|{slot['passage_sentence_count']}|"
        f"{slot['critical_sentence_position']}|{slot['domain']}|{slot['template']['id']}".encode()
    ).hexdigest()[:10]
    return f"raw_{slot['index']:05d}_{fingerprint}"


def _eligible_features(slot: dict[str, Any]) -> list:
    pool = [
        feature for feature in FEATURES
        if feature.layer == slot["layer"]
        and feature.macro == slot["macro_phenomenon"]
        and feature.objective == slot["objective"]
    ]
    if slot["strict_minimal_pair"]:
        pool = [feature for feature in pool if feature.key not in NON_STRICT_FEATURE_KEYS]
    if slot["layer"] == "chain" and slot["target_split"] == "development":
        pool = [feature for feature in pool if feature.key in DEV_COMPOSITION_KEYS]
    elif slot["layer"] == "chain" and slot["generalization_bucket"] == "composition_holdout":
        pool = [feature for feature in pool if feature.key not in DEV_COMPOSITION_KEYS]
    return pool


def _balance_features(slots: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Greedily balance features inside fixed macro/objective slots under holdout constraints."""
    counts = Counter()
    seed = int(cfg["seed"])
    ordered = sorted(
        slots,
        key=lambda slot: (
            len(_eligible_features(slot)),
            _seed(seed, "feature_slot_order", slot["index"]),
        ),
    )
    for slot in ordered:
        eligible = _eligible_features(slot)
        if not eligible:
            raise ValueError(f"{slot['slot_id']} için uygun fenomen yok")
        feature = min(
            eligible,
            key=lambda item: (
                counts[item.key], _seed(seed, "feature_tie", slot["index"], item.key)
            ),
        )
        counts[feature.key] += 1
        slot["feature"] = feature.to_dict()
        slot["hard_profile"] = hard_profile(feature, slot["family_mode"])
        slot["slot_id"] = _slot_id(slot, feature, cfg)
    return sorted(slots, key=lambda slot: slot["index"])


def _feature_for(
    index: int,
    layer: str,
    macro: str,
    seed: int,
    target_split: str,
    bucket: str,
    strict_minimal_pair: bool,
):
    pool = [feature for feature in FEATURES if feature.layer == layer and feature.macro == macro]
    if strict_minimal_pair:
        pool = [feature for feature in pool if feature.key not in NON_STRICT_FEATURE_KEYS]
    if layer == "chain" and target_split == "development":
        pool = [feature for feature in pool if feature.key in DEV_COMPOSITION_KEYS]
    elif layer == "chain" and bucket == "composition_holdout":
        pool = [feature for feature in pool if feature.key not in DEV_COMPOSITION_KEYS]
    if not pool:
        pool = [feature for feature in FEATURES if feature.layer == layer]
    return _round_robin(pool, index, seed, f"feature:{layer}:{macro}")


def make_slot(index: int, cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg["seed"])
    split_distribution = {"development": 1 / 6, "sealed_test": 5 / 6}
    target_split = _weighted_value(split_distribution, index, 6, seed, "split")
    split_index = _weighted_occurrence_rank(
        split_distribution, target_split, index, 6, seed, "split"
    )
    bucket = _weighted_value(
        cfg["generalization_distribution"], split_index, 10, seed, f"bucket:{target_split}"
    )
    query_sentence_count = int(
        _weighted_value(
            cfg["query_sentence_distribution"], split_index, 4, seed,
            f"query_sentences:{target_split}",
        )
    )
    family_mode = _weighted_value(
        cfg["family_mode_distribution"], split_index, 100, seed,
        f"family_mode:{target_split}",
    )
    query_expression = _weighted_value(
        cfg["query_expression_distribution"], split_index, 2, seed,
        f"query_expression:{target_split}",
    )
    query_gold_lexical_band = _weighted_value(
        cfg["query_gold_lexical_distribution"], split_index, 10, seed,
        f"query_gold_lexical:{target_split}",
    )
    passage_sentence_count = int(
        _weighted_value(
            cfg["passage_sentence_distribution"], split_index, 10, seed,
            f"passage_sentences:{target_split}",
        )
    )
    passage_rank = _weighted_occurrence_rank(
        cfg["passage_sentence_distribution"], str(passage_sentence_count), split_index, 10,
        seed, f"passage_sentences:{target_split}",
    )
    critical_sentence_position = _round_robin(
        list(range(1, passage_sentence_count + 1)), passage_rank, seed,
        f"critical_position:{target_split}:p{passage_sentence_count}",
    )
    strict_minimal_pair = family_mode == "strict_minimal"
    generator_ids = [row["id"] for row in cfg["generation"]["generators"]]
    generator_id = _round_robin(generator_ids, split_index, seed, f"generator:{target_split}")

    if bucket == "composition_holdout":
        layer = "chain"
    else:
        chain_rate = float(cfg["non_composition_chain_rate"])
        layer = _weighted_value(
            {"single": 1.0 - chain_rate, "chain": chain_rate}, index, 4, seed, "layer"
        )

    available_macros = [macro for macro in MACROS if (macro == "suffix_chain_composition") == (layer == "chain")]
    force_allomorph = False
    if layer == "single":
        force_allomorph = _weighted_value(
            {"no": 1.0 - float(cfg["single_allomorph_rate"]), "yes": float(cfg["single_allomorph_rate"])},
            index,
            6,
            seed,
            "allomorph",
        ) == "yes"
    if force_allomorph:
        macro = "derivation_nonfinite_allomorphy"
        allomorphs = [feature for feature in FEATURES if feature.objective == "allomorph_invariance"]
        feature = _round_robin(allomorphs, index, seed, "allomorph_feature")
    else:
        macro = _round_robin(available_macros, index, seed, f"macro:{layer}")
        feature = _feature_for(
            index, layer, macro, seed, target_split, bucket, strict_minimal_pair
        )
        if feature.objective == "allomorph_invariance":
            non_allomorph = [
                candidate for candidate in FEATURES
                if candidate.layer == "single" and candidate.macro == macro
                and candidate.objective != "allomorph_invariance"
            ]
            feature = _round_robin(non_allomorph, index, seed, "non_allomorph_feature")
    domain = _round_robin(DOMAINS, index, seed, "domain")
    register = _round_robin(REGISTERS, index, seed, "register")

    # Template holdout uses a dedicated abstract-template subset. The future train generator must
    # consume the frozen holdout manifest and exclude these IDs, not infer disjointness from prose.
    if bucket == "template_holdout":
        template_pool = TEMPLATES[-4:-2] if target_split == "development" else TEMPLATES[-2:]
    else:
        template_pool = TEMPLATES[:-4]
    template = _round_robin(template_pool, index, seed, f"template:{bucket}")
    domain_shift = _weighted_value(
        {"no": 1.0 - float(cfg["domain_shift_fraction"]), "yes": float(cfg["domain_shift_fraction"])},
        index,
        5,
        seed,
        "domain_shift",
    ) == "yes"
    if target_split == "sealed_test" and domain_shift:
        domain, register = _round_robin(
            SEALED_DOMAIN_REGISTER_PAIRS, index, seed, "sealed_domain_register_pair"
        )
    elif target_split == "development" and (domain, register) in SEALED_DOMAIN_REGISTER_PAIRS:
        register = REGISTERS[(REGISTERS.index(register) + 1) % len(REGISTERS)]

    tags = [bucket]
    if domain_shift:
        tags.append("domain_shift")

    slot = {
        "index": index,
        "slot_id": "pending",
        "target_split": target_split,
        "generalization_bucket": bucket,
        "generalization_tags": tags,
        "objective": feature.objective,
        "layer": layer,
        "macro_phenomenon": feature.macro,
        "feature": feature.to_dict(),
        "domain": domain,
        "register": register,
        "query_sentence_count": query_sentence_count,
        "passage_sentence_count": passage_sentence_count,
        "critical_sentence_position": critical_sentence_position,
        "family_mode": family_mode,
        "query_expression": query_expression,
        "query_gold_lexical_band": query_gold_lexical_band,
        "hard_profile": hard_profile(feature, family_mode),
        "strict_minimal_pair": strict_minimal_pair,
        "generator_id": generator_id,
        "template": dict(template),
        "semantic_frame_id": f"frame_{index:05d}",
        "lemma_policy": "test_only_critical_lemma" if bucket == "lemma_holdout" else "open_critical_lemma",
        "composition_policy": (
            "components_seen_chain_unseen" if bucket == "composition_holdout" else "open_composition"
        ),
    }
    slot["slot_id"] = _slot_id(slot, feature, cfg)
    return slot


def build_plan(cfg: dict[str, Any], size: int | None = None) -> list[dict[str, Any]]:
    full_size = max(final_target(cfg), size or 0)
    offset = int(cfg.get("slot_index_offset", 0))
    slots = _balance_features(
        [make_slot(index, cfg) for index in range(offset, offset + full_size)], cfg
    )
    return slots[:size] if size is not None else slots


def plan_statistics(slots: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = (
        "target_split", "generalization_bucket", "objective", "layer", "macro_phenomenon",
        "domain", "register", "query_sentence_count", "passage_sentence_count",
        "critical_sentence_position",
        "family_mode", "query_expression", "query_gold_lexical_band",
        "strict_minimal_pair", "generator_id",
    )
    return {
        field: dict(sorted(Counter(str(slot[field]) for slot in slots).items())) for field in fields
    }


def plan_hash(slots: list[dict[str, Any]]) -> str:
    payload = json.dumps(slots, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
