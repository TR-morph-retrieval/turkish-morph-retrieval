"""Dependency-free regression tests for the new test-set package."""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from .config import ConfigError, load_config, model_family, validate_config
from .dataset_memory import DatasetMemory, semantic_profile_problems
from .prompts import (
    _blind_candidates,
    build_generation_batch_prompt,
    build_generation_prompt,
    build_local_candidate_repair_prompt,
    build_morphology_judge_prompt,
    build_repair_prompt,
    build_semantic_judge_prompt,
)
from .evaluation import _remove_once, evaluate_run, validate_binary_qrels
from .morphology import _expected_feature_check
from .judge_report import judge_calibration_report
from .planner import build_plan, make_slot, plan_statistics
from .pipeline import (
    _group_slots_by_generator,
    _apply_local_candidate_repairs,
    _process_generation_batch,
    _process_slot,
    _resume_refill_rounds,
)
from .schema import generation_batch_schema, localized_repair_schema
from .providers import ClaudeCliProvider
from .review import apply_human_reviews, export_human_review
from .ranges import range_status, validate_shared_ranges
from .selection import select_balanced
from .taxonomy import FEATURES, FEATURE_BY_KEY, hard_profile
from .validators import (
    _contains_word,
    corpus_problems,
    interpret_morphology_judge,
    interpret_semantic_judges,
    normalize_family,
    validate_family,
)


def _fixture_slot(cfg):
    slot = make_slot(0, cfg)
    slot.update({
        "slot_id": "fixture_neg_0000",
        "target_split": "development",
        "generalization_bucket": "standard",
        "generalization_tags": ["standard"],
        "objective": "morpheme_sensitivity",
        "layer": "single",
        "macro_phenomenon": FEATURE_BY_KEY["NEG"].macro,
        "feature": FEATURE_BY_KEY["NEG"].to_dict(),
        "domain": "daily_life",
        "register": "everyday",
        "query_sentence_count": 1,
        "passage_sentence_count": 1,
        "critical_sentence_position": 1,
        "family_mode": "controlled_diverse",
        "query_expression": "morph_explicit",
        "query_gold_lexical_band": "low",
        "hard_profile": hard_profile(FEATURE_BY_KEY["NEG"], "controlled_diverse"),
        "template": {"id": "event_report", "description": "doğal bir olay bildirimi"},
        "semantic_frame_id": "frame_fixture",
        "strict_minimal_pair": False,
        "generator_id": "generator_a",
    })
    return slot


def _fixture_raw():
    candidates = [
        ("positive_01", "positive", "equivalence_positive", "Ece raporu toplantıdan önce bitirmedi.", "bitirmedi", "target_preserved"),
        ("hard_01", "hard_negative", "minimal_morph_negative", "Ece raporu toplantıdan önce tamamladı.", "tamamladı", "feature_changed"),
        ("hard_02", "hard_negative", "same_lemma_wrong_inflection", "Ece raporu toplantıdan önce tamamlayacak.", "tamamlayacak", "wrong_inflection"),
        ("hard_03", "hard_negative", "related_feature_negative", "Ece raporu toplantıdan sonra tamamlamadı.", "tamamlamadı", "wrong_inflection"),
        ("hard_04", "hard_negative", "same_morph_wrong_content", "Ece sunumu toplantıdan önce tamamlamadı.", "tamamlamadı", "same_feature_wrong_content"),
        ("hard_05", "hard_negative", "state_participant_time_trap", "Mert raporu toplantıdan önce tamamlamadı.", "tamamlamadı", "target_preserved"),
        ("hard_06", "hard_negative", "close_paraphrase_wrong_meaning", "Ece raporu toplantıdan önce başlatmadı.", "başlatmadı", "feature_changed"),
        ("hard_07", "hard_negative", "argument_role_reversal", "Rapor Ece'yi toplantıdan önce tamamlamadı.", "tamamlamadı", "target_preserved"),
        ("hard_08", "hard_negative", "morph_distractor", "Ece toplantıyı bitirmedi ama rapor tamamlandı.", "bitirmedi", "feature_changed"),
        ("easy_01", "easy_negative", "easy_negative", "Mert otobüs biletini sabah erkenden aldı.", "aldı", "same_domain_off_intent"),
        ("easy_02", "easy_negative", "easy_negative", "Derya bahçedeki çiçekleri akşam suladı.", "suladı", "same_domain_off_intent"),
    ]
    return {
        "semantic_frame_id": "frame_fixture",
        "semantic_profile": {
            "narrative_tag": "rapor_teslimi",
            "event_type": "belge_tamamlama",
            "participant_roles": ["agent", "theme"],
            "participant_bindings": [
                {"role": "agent", "participant": "Ece"},
                {"role": "theme", "participant": "rapor"},
            ],
            "polarity": "negative",
            "temporal_frame": "past",
            "scope_target": "predicate",
        },
        "template_id": "event_report",
        "critical_lemma": "tamamlamak",
        "critical_word_query": "tamamlamadı",
        "critical_word_positive": "bitirmedi",
        "feature_delta": "NEG korunur; minimal negatifte NEG kaldırılır",
        "edit_script": {
            "applies": False, "positive_form": "", "minimal_negative_form": "",
            "operation": "", "changed_feature": "", "invariants": [],
        },
        "query": "Toplantı başlamadan Ece raporu hâlâ tamamlamadı.",
        "context_sentences": [],
        "candidates": [
            {"candidate_slot": candidate_slot, "role": role, "subtype": subtype,
             "critical_sentence": text, "critical_word": word,
             "morph_relation": relation, "reason": subtype}
            for candidate_slot, role, subtype, text, word, relation in candidates
        ],
        "generation_notes": "fixture",
    }


def run() -> list[str]:
    failures = []
    if semantic_profile_problems(_fixture_raw()["semantic_profile"]):
        failures.append("geçerli semantic_profile reddedildi")
    bad_profile = {**_fixture_raw()["semantic_profile"], "narrative_tag": "Ham cümle etiketi"}
    if not semantic_profile_problems(bad_profile):
        failures.append("serbest metin semantic_profile etiketi reddedilmedi")
    bad_bindings = deepcopy(_fixture_raw()["semantic_profile"])
    bad_bindings["participant_bindings"] = [
        {"role": "agent", "participant": "Ece"},
    ]
    if not semantic_profile_problems(bad_bindings):
        failures.append("participant role/binding uyuşmazlığı reddedilmedi")
    if not _expected_feature_check("ALLO.ACC", {"ufeats": "Case=Acc|Number=Sing"}):
        failures.append("allomorph query UFeats eşlemesi çalışmadı")
    if not _expected_feature_check(
        "POSS.2PL", {"ufeats": "Person[psor]=2|Number[psor]=Plur|Case=Gen"}
    ):
        failures.append("POSS.2PL query UFeats eşlemesi çalışmadı")
    metric_summary, metric_rows = evaluate_run(
        {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
        {"q1": ["x", "d1"], "q2": ["d2", "x"]},
    )
    if metric_summary.get("recall@1") != 0.5 or metric_summary.get("mrr@10") != 0.75:
        failures.append(f"closed retrieval metrikleri yanlış: {metric_summary}")
    if "map@10" in metric_summary or "bpref" in metric_summary:
        failures.append("tek-gold düzende gereksiz MAP@10/bpref hâlâ raporlanıyor")
    if metric_summary.get("mean_rank") != 1.5 or len(metric_rows) != 2:
        failures.append(f"rank özeti yanlış: {metric_summary}")
    condensed, condensed_rows = evaluate_run(
        {"q1": {"d1": 1.0, "n1": 0.0}}, {"q1": ["unjudged", "d1", "n1"]},
        unjudged_policy="condensed",
    )
    if condensed.get("recall@1") != 1.0 or condensed_rows[0].get("judged@1") != 0.0:
        failures.append(f"unjudged condensed evaluation yanlış: {condensed} / {condensed_rows}")
    try:
        validate_binary_qrels({"q1": {"d1": 2.0, "n1": 0.0}})
        failures.append("binary qrels relevance=2 değerini reddetmedi")
    except ValueError:
        pass

    # Regression: _contains_word must match token sequences, not arbitrary substrings
    if _contains_word("fiyatları hızla arttı", "at"):
        failures.append("_contains_word 'fiyat' içindeki 'at' alt-dizesini hatalı kabul etti")
    if not _contains_word("topu bana doğru at", "at"):
        failures.append("_contains_word tam belirteç olan 'at' sözcüğünü bulamadı")
    if not _contains_word("keşke haberi duyurmuş olsaydı", "duyurmuş olsaydı"):
        failures.append("_contains_word çoklu belirteç öbeğini bulamadı")

    # Regression: _remove_once must respect Turkish word boundaries during ablation
    ablated_sample = _remove_once("Ahmet balkonda oturdu ve kalemi al.", "al")
    if ablated_sample != "Ahmet balkonda oturdu ve kalemi [MASK].":
        failures.append(f"_remove_once sözcük sınırını ihlal etti: {ablated_sample}")

    # Regression: evaluate_run truncated rankings must report retrieved-only statistics
    trunc_summary, _ = evaluate_run(
        {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
        {"q1": ["d1", "x"], "q2": ["other1", "other2"]},
    )
    if "mean_rank" in trunc_summary or trunc_summary.get("mean_rank_retrieved") != 1.0 or trunc_summary.get("retrieved_ratio") != 0.5:
        failures.append(f"truncated ranking metrikleri yanlış: {trunc_summary}")

    cfg = load_config(runtime=False)
    slots_600 = build_plan(cfg, 600)
    if build_plan(cfg) != slots_600:
        failures.append("varsayılan plan doğrudan 600 kota slotu üretmiyor")
    with TemporaryDirectory() as temporary:
        empty_ranges = validate_shared_ranges(temporary)
        if empty_ranges["coverage"] != {"codex": 0, "claude": 0}:
            failures.append(f"boş shared range coverage yanlış: {empty_ranges}")
        first_range = range_status("codex", 1, 50, temporary)
        if first_range["count"] != 50 or first_range["offset_internal"] != 0:
            failures.append(f"1-based range dönüşümü yanlış: {first_range}")
        try:
            range_status("codex", 50, 120, temporary)
            failures.append("ilk shared range 1 yerine 50'den başlayabildi")
        except ValueError:
            pass
        try:
            range_status("claude", 1, 10, temporary)
            failures.append("Claude sırası 300 Codex family tamamlanmadan başlayabildi")
        except ValueError:
            pass
    stats = plan_statistics(slots_600)
    split_counts = Counter(slot["target_split"] for slot in slots_600)
    if split_counts != {"development": 100, "sealed_test": 500}:
        failures.append(f"600 slot split kotası yanlış: {split_counts}")
    planned_features = Counter(slot["feature"]["key"] for slot in slots_600)
    if len(FEATURES) != 76 or set(planned_features) != {feature.key for feature in FEATURES}:
        failures.append(
            f"76 fenomenin tamamı planda değil: taxonomy={len(FEATURES)}, "
            f"planned={len(planned_features)}"
        )
    new_features = {
        "COP.NEG", "COP.TAM", "Q.PART.SCOPE", "NMLZ.MA_VS_DIK",
        "REL.GEN.POSS", "ANAPHOR.AGR",
        "MORPH.CONTEXT_AMBIG", "DERIV.IG_CHAIN", "CASE.ROLE.FRAME", "SUSP.AFFIX",
        "MWE.MORPH",
    }
    if not new_features <= set(planned_features):
        failures.append(f"yeni fenomenler eksik: {sorted(new_features - set(planned_features))}")
    balance_groups: dict[tuple[str, str], list[int]] = {}
    for key, count in planned_features.items():
        feature = FEATURE_BY_KEY[key]
        balance_groups.setdefault((feature.macro, feature.objective), []).append(count)
    unbalanced = {
        group: values for group, values in balance_groups.items()
        if max(values) - min(values) > 1
    }
    if unbalanced:
        failures.append(f"macro/objective içi fenomen dağılımı dengesiz: {unbalanced}")
    if any(
        slot["strict_minimal_pair"] and slot["feature"]["key"] in {
            "Q.PART.SCOPE", "MORPH.CONTEXT_AMBIG", "SUSP.AFFIX", "MWE.MORPH",
        }
        for slot in slots_600
    ):
        failures.append("strict olmayan bağlam/koordinasyon fenomeni minimal-pair slice'ına girdi")
    expected = {
        "target_split": {"development": 100, "sealed_test": 500},
        "query_sentence_count": {"1": 450, "2": 150},
        "passage_sentence_count": {"1": 180, "2": 180, "3": 180, "4": 60},
        "generalization_bucket": {
            "composition_holdout": 120, "lemma_holdout": 120,
            "standard": 240, "template_holdout": 120,
        },
        "strict_minimal_pair": {"False": 450, "True": 150},
        "family_mode": {
            "controlled_diverse": 270, "natural_retrieval": 180, "strict_minimal": 150,
        },
        "query_expression": {"morph_explicit": 300, "semantic_paraphrase": 300},
        "query_gold_lexical_band": {"high": 180, "low": 180, "medium": 240},
        "generator_id": {"generator_a": 300, "generator_b": 300},
    }
    for field, wanted in expected.items():
        if stats[field] != wanted:
            failures.append(f"plan {field}: {stats[field]} != {wanted}")
    if build_plan(cfg, 50) != build_plan(cfg, 600)[:50]:
        failures.append("planner prefix-stable değil")
    resumed = _resume_refill_rounds([
        {"slot_id": "a", "next_refill_round": 3},
        {"slot_id": "a", "next_refill_round": 6},
        {"slot_id": "b", "next_refill_round": 3},
    ])
    if resumed != {"a": 6, "b": 3}:
        failures.append(f"refill resume turu yanlış: {resumed}")
    dev_templates = {slot["template"]["id"] for slot in slots_600 if slot["target_split"] == "development"}
    test_holdout_templates = {
        slot["template"]["id"] for slot in slots_600
        if slot["target_split"] == "sealed_test" and slot["generalization_bucket"] == "template_holdout"
    }
    if dev_templates & test_holdout_templates:
        failures.append("sealed template holdout development planına sızdı")
    dev_chains = {
        slot["feature"]["key"] for slot in slots_600
        if slot["target_split"] == "development" and slot["layer"] == "chain"
    }
    test_holdout_chains = {
        slot["feature"]["key"] for slot in slots_600
        if slot["target_split"] == "sealed_test" and slot["generalization_bucket"] == "composition_holdout"
    }
    if dev_chains & test_holdout_chains:
        failures.append("sealed composition holdout development planına sızdı")
    dev_domain_pairs = {
        (slot["domain"], slot["register"]) for slot in slots_600
        if slot["target_split"] == "development"
    }
    test_shift_pairs = {
        (slot["domain"], slot["register"]) for slot in slots_600
        if slot["target_split"] == "sealed_test" and "domain_shift" in slot["generalization_tags"]
    }
    if dev_domain_pairs & test_shift_pairs:
        failures.append("sealed domain/register shift development planına sızdı")

    fake_items = []
    for slot_item in build_plan(cfg):
        fake_items.append({
            "family_id": f"fake_{slot_item['slot_id']}",
            "slot_id": slot_item["slot_id"],
            "target_split": slot_item["target_split"],
            "generalization_bucket": slot_item["generalization_bucket"],
            "query_sentence_count": slot_item["query_sentence_count"],
            "passage_sentence_count": slot_item["passage_sentence_count"],
            "critical_sentence_position": slot_item["critical_sentence_position"],
            "family_mode": slot_item["family_mode"],
            "query_expression": slot_item["query_expression"],
            "query_gold_lexical_band": slot_item["query_gold_lexical_band"],
            "macro_phenomenon": slot_item["macro_phenomenon"],
            "target_feature": slot_item["feature"]["key"],
            "critical_lemma": f"lemma_{slot_item['index']}",
            "strict_minimal_pair": slot_item["strict_minimal_pair"],
            "generator_id": slot_item["generator_id"],
            "qc": {"quality_score": 1.0},
        })
    try:
        final_selection = select_balanced(
            fake_items, cfg, {"development": 100, "sealed_test": 500}
        )
        if len(final_selection) != 600:
            failures.append("600-family final balanced selection sayısı yanlış")
        for split, query_expected, passage_expected in (
            ("development", {1: 75, 2: 25}, {1: 30, 2: 30, 3: 30, 4: 10}),
            ("sealed_test", {1: 375, 2: 125}, {1: 150, 2: 150, 3: 150, 4: 50}),
        ):
            split_items = [item for item in final_selection if item["target_split"] == split]
            query_actual = dict(Counter(item["query_sentence_count"] for item in split_items))
            passage_actual = dict(Counter(item["passage_sentence_count"] for item in split_items))
            if query_actual != query_expected:
                failures.append(f"{split} query dağılımı yanlış: {query_actual}")
            if passage_actual != passage_expected:
                failures.append(f"{split} passage dağılımı yanlış: {passage_actual}")
            strict_expected = 25 if split == "development" else 125
            if sum(item["strict_minimal_pair"] for item in split_items) != strict_expected:
                failures.append(f"{split} strict minimal-pair kotası yanlış")
            generator_expected = (
                {"generator_a": 50, "generator_b": 50} if split == "development"
                else {"generator_a": 250, "generator_b": 250}
            )
            if dict(Counter(item["generator_id"] for item in split_items)) != generator_expected:
                failures.append(f"{split} generator dengesi yanlış")
            for passage_count in (2, 3, 4):
                position_counts = Counter(
                    item["critical_sentence_position"] for item in split_items
                    if item["passage_sentence_count"] == passage_count
                )
                if len(position_counts) != passage_count or (
                    max(position_counts.values()) - min(position_counts.values()) > 2
                ):
                    failures.append(
                        f"{split} p{passage_count} kritik konum dağılımı dengesiz: "
                        f"{dict(position_counts)}"
                    )
    except Exception as exc:
        failures.append(f"balanced selection başarısız: {type(exc).__name__}: {exc}")

    slot = _fixture_slot(cfg)
    family = normalize_family(_fixture_raw(), slot)
    problems = validate_family(family, slot, cfg)
    if problems:
        failures.append(f"geçerli fixture reddedildi: {problems}")
    with TemporaryDirectory() as temporary:
        memory = DatasetMemory(Path(temporary) / "dataset_memory.sqlite3")
        second_slot = deepcopy(slot)
        second_slot.update({"slot_id": "fixture_neg_0001", "index": 1})
        concurrent_slot = deepcopy(slot)
        concurrent_slot.update({"slot_id": "fixture_neg_0002", "index": 2})
        memory.sync_plan([slot, second_slot, concurrent_slot])
        if not memory.reserve_slot(slot["slot_id"], "worker_a"):
            failures.append("dataset memory planlanmış slotu rezerve edemedi")
        if memory.reserve_slot(slot["slot_id"], "worker_b"):
            failures.append("dataset memory aynı slotu iki workera verdi")
        with ThreadPoolExecutor(max_workers=8) as executor:
            reservations = list(executor.map(
                lambda worker: memory.reserve_slot(concurrent_slot["slot_id"], worker),
                [f"parallel_{index}" for index in range(8)],
            ))
        if sum(reservations) != 1:
            failures.append(f"dataset memory atomik rezervasyon sayısı yanlış: {reservations}")
        family["source_type"] = "llm_generated_cascade_judged"
        memory.record_outcome(slot["slot_id"], "accepted", family, actor="worker_a")
        external = deepcopy(family)
        external.update({
            "family_id": "external_train_fixture",
            "slot_id": "external_train_slot",
            "target_split": "train",
        })
        memory.ingest_families([external], "train_fixture")
        for bucket, expected_text in (
            ("lemma_holdout", "lemma_holdout"),
            ("template_holdout", "template_holdout"),
            ("composition_holdout", "composition_holdout"),
        ):
            probe = {**family, "family_id": f"probe_{bucket}", "generalization_bucket": bucket}
            if not any(expected_text in problem for problem in memory.conflicts_for(probe)):
                failures.append(f"dataset memory {bucket} external çakışmasını yakalamadı")
        context = memory.generation_context(second_slot)
        if family["critical_lemma"] not in context["avoid_critical_lemmas"]:
            failures.append("dataset memory kabul edilen lemmayı sonraki bağlama taşımadı")
        if family["semantic_profile"]["narrative_tag"] not in context["avoid_narrative_tags"]:
            failures.append("dataset memory anlatı etiketini sonraki bağlama taşımadı")
        if family["query"] in json.dumps(context, ensure_ascii=False):
            failures.append("dataset memory ham örneği generation context'e sızdırdı")
        report = memory.report()
        if report["slot_states"] != {"accepted": 1, "planned": 1, "reserved": 1}:
            failures.append(f"dataset memory slot durumları yanlış: {report['slot_states']}")
        drifted = deepcopy(second_slot)
        drifted["domain"] = "finance"
        try:
            memory.sync_plan([drifted])
            failures.append("dataset memory slot contract drift'ini reddetmedi")
        except ValueError:
            pass
    copied_gold_raw = _fixture_raw()
    copied_gold_raw["query"] = "Ece raporu toplantıdan önce tamamlamadı."
    copied_gold_raw["critical_word_query"] = "tamamlamadı"
    copied_gold = normalize_family(copied_gold_raw, slot)
    copied_problems = validate_family(copied_gold, slot, cfg)
    if not any("query ile gold tek-kelime/yakın-kopya" in p for p in copied_problems):
        failures.append("query–gold yakın-kopya koruması çalışmadı")
    hard = {
        candidate["candidate_slot"]: candidate["subtype"]
        for candidate in family["candidates"] if candidate["role"] == "hard_negative"
    }
    expected_hard = {item["slot"]: item["subtype"] for item in slot["hard_profile"]}
    if hard != expected_hard:
        failures.append("fixture uyarlanabilir sekiz hard slotu kapsamıyor")
    if Counter(family["qrels"].values()) != {0: 10, 1: 1}:
        failures.append("family oluşturulurken 1 gold + 10 negative qrels üretilmedi")

    strict_slot = deepcopy(slot)
    strict_slot["family_mode"] = "strict_minimal"
    strict_slot["strict_minimal_pair"] = True
    strict_slot["hard_profile"] = hard_profile(FEATURE_BY_KEY["NEG"], "strict_minimal")
    strict_raw = _fixture_raw()
    strict_raw["critical_word_positive"] = "bitirmedi"
    positive_raw = next(row for row in strict_raw["candidates"] if row["candidate_slot"] == "positive_01")
    minimal_raw = next(row for row in strict_raw["candidates"] if row["candidate_slot"] == "hard_01")
    positive_raw.update({
        "critical_sentence": "Ece raporu toplantıdan önce bitirmedi.",
        "critical_word": "bitirmedi",
    })
    minimal_raw.update({
        "critical_sentence": "Ece raporu toplantıdan önce bitirdi.",
        "critical_word": "bitirdi",
    })
    strict_raw["edit_script"] = {
        "applies": True, "positive_form": "bitirmedi",
        "minimal_negative_form": "bitirdi", "operation": "NEG ekini kaldır",
        "changed_feature": "NEG", "invariants": ["lemma", "token_order", "event"],
    }
    strict_family = normalize_family(strict_raw, strict_slot)
    strict_problems = validate_family(strict_family, strict_slot, cfg)
    if strict_problems:
        failures.append(f"geçerli strict minimal pair reddedildi: {strict_problems}")

    long_slot = deepcopy(slot)
    long_slot.update({
        "query_sentence_count": 2,
        "passage_sentence_count": 4,
        "critical_sentence_position": 3,
    })
    long_raw = _fixture_raw()
    long_raw["query"] = (
        "Toplantı başlamadan Ece raporu hâlâ tamamlamadı. "
        "Kayıt, teslimin hâlâ eksik olduğunu belirtiyor."
    )
    long_raw["context_sentences"] = [
        "Ekip sabah erkenden ofiste toplandı.",
        "Toplantı için bütün hazırlıklar gözden geçirildi.",
        "Yönetici günün sonunda kısa bir değerlendirme yaptı.",
    ]
    long_family = normalize_family(long_raw, long_slot)
    long_problems = validate_family(long_family, long_slot, cfg)
    if long_problems:
        failures.append(f"dört cümleli fixture reddedildi: {long_problems}")
    elif any(candidate["text"] != " ".join([
        *long_raw["context_sentences"][:2],
        candidate["critical_sentence"],
        *long_raw["context_sentences"][2:],
    ]) for candidate in long_family["candidates"]):
        failures.append("kritik cümle planlanan üçüncü konuma yerleşmedi")

    morphology_matches = []
    for candidate in family["candidates"]:
        supports = candidate["role"] == "positive"
        if supports:
            morphology_matches.append(candidate["id"])
    semantic_judge = {
        "fully_relevant_candidate_ids": [family["gold_id"]],
        "unnatural_candidate_ids": [], "internally_inconsistent_candidate_ids": [],
        "length_or_style_artifact": False, "family_naturalness": 5,
        "confidence": 95, "abstain": False, "notes": "fixture",
    }
    morphology_judge = {
        "target_matching_candidate_ids": morphology_matches,
        "morphologically_invalid_candidate_ids": [], "unclear_candidate_ids": [],
        "allomorph_treated_as_wrong": False, "confidence": 95, "abstain": False,
        "notes": "fixture",
    }
    semantic_problems, _ = interpret_semantic_judges(
        family, [semantic_judge, deepcopy(semantic_judge)], cfg
    )
    morphology_problems, _ = interpret_morphology_judge(family, morphology_judge, cfg)
    if semantic_problems or morphology_problems:
        failures.append(
            f"geçerli cascade judge fixture reddedildi: {semantic_problems + morphology_problems}"
        )
    bad_support = deepcopy(semantic_judge)
    negative = next(c["id"] for c in family["candidates"] if c["role"] != "positive")
    bad_support["fully_relevant_candidate_ids"].append(negative)
    support_problems, _ = interpret_semantic_judges(
        family, [bad_support, deepcopy(semantic_judge)], cfg
    )
    if not any("query desteği" in problem for problem in support_problems):
        failures.append("judge false-negative/query desteği kontrolü çalışmadı")
    bad_consistency = deepcopy(semantic_judge)
    bad_consistency["internally_inconsistent_candidate_ids"] = [family["gold_id"]]
    consistency_problems, _ = interpret_semantic_judges(
        family, [bad_consistency, deepcopy(bad_consistency)], cfg
    )
    if not any("iç tutarsız" in problem for problem in consistency_problems):
        failures.append("judge iç tutarlılık kontrolü çalışmadı")
    low_naturalness = deepcopy(semantic_judge)
    low_naturalness["unnatural_candidate_ids"] = [family["gold_id"]]
    naturalness_problems, _ = interpret_semantic_judges(
        family, [low_naturalness, deepcopy(low_naturalness)], cfg
    )
    if not any("doğal olmayan aday" in problem for problem in naturalness_problems):
        failures.append("candidate-level naturalness kapısı çalışmadı")
    low_family_naturalness = deepcopy(semantic_judge)
    low_family_naturalness["family_naturalness"] = 2
    family_naturalness_problems, _ = interpret_semantic_judges(
        family, [low_family_naturalness], cfg
    )
    if not any("family naturalness" in problem for problem in family_naturalness_problems):
        failures.append("family-level naturalness kapısı çalışmadı")
    uncertain_semantic = deepcopy(bad_support)
    uncertain_semantic["confidence"] = 60
    uncertain_problems, _ = interpret_semantic_judges(family, [uncertain_semantic], cfg)
    if uncertain_problems:
        failures.append("düşük güvenli semantic judge veto kullandı")
    uncertain_morphology = deepcopy(morphology_judge)
    uncertain_morphology["confidence"] = 60
    uncertain_morphology["unclear_candidate_ids"] = [family["gold_id"]]
    uncertain_morphology_problems, _ = interpret_morphology_judge(
        family, uncertain_morphology, cfg
    )
    if uncertain_morphology_problems:
        failures.append("düşük güvenli morphology judge veto kullandı")
    advisory_morphology = deepcopy(morphology_judge)
    advisory_morphology["unclear_candidate_ids"] = [family["gold_id"]]
    advisory_morphology_problems, _ = interpret_morphology_judge(
        family, advisory_morphology, cfg
    )
    if advisory_morphology_problems:
        failures.append("morphology belirsizliği veto kullandı")

    def response(data, number):
        return SimpleNamespace(
            data=deepcopy(data), provider="fake", model="fake/model",
            request_hash=f"request-{number}", cache_hit=False, usage={},
            actual_model="fake/model", route_provider="fake-route",
        )

    class RefillGenerator:
        def __init__(self, invalid_calls=0):
            self.calls = 0
            self.invalid_calls = invalid_calls

        def call_json(self, _system, _prompt, schema, _task):
            self.calls += 1
            if self.calls <= self.invalid_calls:
                return response({}, self.calls)
            if schema.get("name") == "turkish_morph_candidate_repair":
                wanted = set(
                    schema["schema"]["properties"]["candidates"]["items"]
                    ["properties"]["candidate_slot"]["enum"]
                )
                patches = [
                    row for row in _fixture_raw()["candidates"]
                    if row["candidate_slot"] in wanted
                ]
                return response({"candidates": patches}, self.calls)
            return response(_fixture_raw(), self.calls)

    class SemanticJudge:
        def __init__(self, rejected_calls=0):
            self.calls = 0
            self.rejected_calls = rejected_calls

        def call_json(self, *_args):
            self.calls += 1
            verdict = deepcopy(semantic_judge)
            if self.calls <= self.rejected_calls:
                verdict["fully_relevant_candidate_ids"] = []
            return response(verdict, self.calls)

    class MorphologyJudge:
        def call_json(self, *_args):
            return response(morphology_judge, 100)

    generator = RefillGenerator(invalid_calls=2)
    status, refilled = _process_slot(
        slot, cfg, {"generator_a": generator},
        {"semantic": SemanticJudge(), "morphology": MorphologyJudge()},
        start_refill_round=0,
    )
    if status != "accepted" or refilled["provenance"]["refill_round"] != 1:
        failures.append("deterministic QC reddinden sonra taze refill çalışmadı")
    generator = RefillGenerator()
    status, refilled = _process_slot(
        slot, cfg, {"generator_a": generator},
        {"semantic": SemanticJudge(rejected_calls=1), "morphology": MorphologyJudge()},
        start_refill_round=4,
    )
    if status != "accepted" or generator.calls != 2 or refilled["provenance"]["refill_round"] != 5:
        failures.append(
            "judge reddinden sonra taze refill çalışmadı: "
            f"status={status}, calls={generator.calls}, "
            f"round={refilled.get('provenance', {}).get('refill_round')}, "
            f"attempts={refilled.get('provenance', {}).get('generator_attempts')}, "
            f"history={refilled.get('provenance', {}).get('rejected_replacements')}"
        )
    low_confidence_semantic = deepcopy(semantic_judge)
    low_confidence_semantic["confidence"] = 60
    low_confidence_semantic["abstain"] = True

    class LowConfidenceSemanticJudge:
        def call_json(self, *_args):
            return response(low_confidence_semantic, 201)

    status, prioritized = _process_slot(
        slot, cfg, {"generator_a": RefillGenerator()},
        {"semantic": LowConfidenceSemanticJudge(), "morphology": MorphologyJudge()},
        start_refill_round=8,
    )
    if status != "accepted" or not prioritized.get("qc", {}).get("human_review_priority"):
        failures.append("düşük güvenli kabul human_review_priority almadı")
    if not any(
        "semantic_1:abstain" == reason
        for reason in prioritized.get("qc", {}).get("human_review_priority_reasons", [])
    ):
        failures.append("human_review_priority nedeni kaydedilmedi")

    class FixedSemanticJudge:
        def __init__(self, verdict):
            self.verdict = verdict

        def call_json(self, *_args):
            return response(self.verdict, 202)

    clean_mid_confidence = deepcopy(semantic_judge)
    clean_mid_confidence["confidence"] = 80
    status, clean_mid = _process_slot(
        slot, cfg, {"generator_a": RefillGenerator()},
        {"semantic": FixedSemanticJudge(clean_mid_confidence), "morphology": MorphologyJudge()},
        start_refill_round=9,
    )
    if status != "accepted" or clean_mid.get("qc", {}).get("human_review_priority"):
        failures.append("temiz 60–84 semantic kararı gereksiz human-review priority aldı")

    advisory_mid_confidence = deepcopy(clean_mid_confidence)
    advisory_mid_confidence["unnatural_candidate_ids"] = [family["gold_id"]]
    status, advisory_mid = _process_slot(
        slot, cfg, {"generator_a": RefillGenerator()},
        {"semantic": FixedSemanticJudge(advisory_mid_confidence), "morphology": MorphologyJudge()},
        start_refill_round=10,
    )
    if status != "accepted" or not advisory_mid.get("qc", {}).get("human_review_priority"):
        failures.append("uyarılı 60–84 semantic kararı human-review priority almadı")

    collision_items = [
        {
            "family_id": "collision_a", "semantic_frame_id": "frame_collision_a",
            "query": "Doktor ilk hastanın kontrolünü tamamladı.",
            "candidates": [
                {"id": "a_gold", "role": "positive", "text": "Doktor ilk hastayı muayene etti."},
                {"id": "a_easy", "role": "easy_negative", "text": "Klinik öğleden sonra kapandı."},
            ],
        },
        {
            "family_id": "collision_b", "semantic_frame_id": "frame_collision_b",
            "query": "İkinci hastanın muayenesini yapan kişiyi bildirir.",
            "candidates": [
                {"id": "b_gold", "role": "positive", "text": "Başhekim ikinci hastayı muayene etti."},
                {"id": "b_easy", "role": "easy_negative", "text": "Doktor ilk hastayı muayene etti."},
            ],
        },
    ]
    if not any(
        "gold'u easy" in problem
        for problem in corpus_problems(collision_items, cfg)
    ):
        failures.append("cross-family easy→gold çakışma kapısı çalışmadı")

    allomorph_slot = deepcopy(slot)
    allomorph_slot["feature"] = FEATURE_BY_KEY["ALLO.LOC"].to_dict()
    allomorph_slot["objective"] = "allomorph_invariance"
    allomorph_slot["hard_profile"] = hard_profile(
        FEATURE_BY_KEY["ALLO.LOC"], allomorph_slot["family_mode"]
    )
    allomorph_family = normalize_family(_fixture_raw(), allomorph_slot)
    positive = next(c for c in allomorph_family["candidates"] if c["role"] == "positive")
    positive["morph_relation"] = "allomorph_equivalent"
    negative = next(c for c in allomorph_family["candidates"] if c["role"] == "hard_negative")
    negative["morph_relation"] = "allomorph_equivalent"
    if not any("geçerli allomorph negatif" in p for p in validate_family(allomorph_family, allomorph_slot, cfg)):
        failures.append("allomorph-negatif koruması çalışmadı")

    failures.extend(_check_judge_blindness(cfg))
    failures.extend(_check_model_family_gates())
    failures.extend(_check_claude_cli_adapter())
    failures.extend(_check_refill_round_nonce(cfg))
    failures.extend(_check_generation_batching(cfg))
    failures.extend(_check_human_review_roundtrip(cfg, slot, family))
    return failures


def _check_human_review_roundtrip(cfg, slot, family) -> list[str]:
    failures = []
    family = deepcopy(family)
    family.setdefault("qc", {})["human_review_priority"] = True
    family["qc"]["human_review_priority_reasons"] = ["semantic_1:abstain"]
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = SimpleNamespace(
            root=root,
            accepted=root / "accepted.jsonl",
            rejected=root / "rejected.jsonl",
            memory=root / "dataset_memory.sqlite3",
        )
        memory = DatasetMemory(paths.memory)
        memory.sync_plan([slot])
        memory.record_outcome(slot["slot_id"], "accepted", family, actor="selftest")
        paths.accepted.write_text(json.dumps(family, ensure_ascii=False) + "\n", encoding="utf-8")
        decisions = root / "incoming.jsonl"
        decision_rows = [
            {
                "slot_id": slot["slot_id"], "reviewer_id": reviewer,
                "answers_query": [family["gold_id"]], "morphology_valid": True,
                "naturalness_valid": True, "decision": "accept", "notes": "fixture",
            }
            for reviewer in ("reviewer_a", "reviewer_b")
        ]
        decisions.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decision_rows),
            encoding="utf-8",
        )
        with patch("test.review.paths_for", return_value=paths), \
             patch("test.review.current_accepted", return_value=[family]), \
             patch("test.review.select_balanced", return_value=[family]):
            exported = export_human_review("fixture_review")
            manifest = json.loads(Path(exported["manifest"]).read_text(encoding="utf-8"))
            if any("target_feature" in row for row in manifest["semantic_items"]):
                failures.append("human semantic review manifesti target_feature sızdırıyor")
            if exported.get("priority") != 1 or manifest.get("priority_count") != 1:
                failures.append("human review priority sayacı çalışmadı")
            if not manifest["semantic_items"][0].get("human_review_priority"):
                failures.append("human review priority etiketi manifeste taşınmadı")
            if "human_review_priority_reasons" in manifest["semantic_items"][0]:
                failures.append("human review priority nedeni kör manifeste sızdı")
            applied = apply_human_reviews("fixture_review", decisions)
        with patch("test.judge_report.paths_for", return_value=paths):
            calibration = judge_calibration_report("fixture_review")
        if applied["resolved"] != {"accepted": 1}:
            failures.append(f"human review consensus accepted uygulamadı: {applied}")
        if memory.slot_status(slot["slot_id"]) != "accepted" or not paths.accepted.exists():
            failures.append("human review accepted sonucu registry/JSONL'e yazılmadı")
        if calibration["human_reviewed_slots"] != 1:
            failures.append(f"judge calibration human review'u saymadı: {calibration}")
    return failures


def _check_refill_round_nonce(cfg) -> list[str]:
    """A replacement round must produce a different prompt, or the provider cache (keyed on the
    prompt hash) replays the round that already failed and the refill loop can never recover."""
    failures = []
    slot = _fixture_slot(cfg)
    prompts = [build_generation_prompt({**slot, "refill_round": r}) for r in range(3)]
    if len(set(prompts)) != 3:
        failures.append("refill_round nonce prompt'a ulaşmıyor; replacement turları cache tekrarı")
    if prompts[0] != build_generation_prompt(slot):
        failures.append("refill_round=0 prompt'u nonce'suz hâlinden farklı; mevcut cache boşa düşer")
    repairs = [build_repair_prompt({**slot, "refill_round": r}, {}, ["x"]) for r in range(3)]
    if len(set(repairs)) != 3:
        failures.append("repair prompt'u refill turları arasında ayrışmıyor")
    previous = _fixture_raw()
    candidate_slot = "hard_01"
    patch_payload = {"candidates": [{
        "candidate_slot": candidate_slot,
        "critical_sentence": "Ece bugün raporu arşive bırakmadı.",
        "critical_word": "bırakmadı",
    }]}
    repaired = _apply_local_candidate_repairs(previous, patch_payload, [candidate_slot])
    before_others = [row for row in previous["candidates"] if row["candidate_slot"] != candidate_slot]
    after_others = [row for row in repaired["candidates"] if row["candidate_slot"] != candidate_slot]
    if before_others != after_others or previous["query"] != repaired["query"]:
        failures.append("local candidate repair sağlam family alanlarını değiştirdi")
    changed = next(row for row in repaired["candidates"] if row["candidate_slot"] == candidate_slot)
    if changed["critical_word"] != "bırakmadı":
        failures.append("local candidate repair hedef slotu güncellemedi")
    local_schema = localized_repair_schema([candidate_slot])
    local_prompt = build_local_candidate_repair_prompt(
        slot, previous, ["fixture candidate hatası"], [candidate_slot]
    )
    if local_schema["schema"]["properties"]["candidates"]["maxItems"] != 1:
        failures.append("local repair şeması tek adayla sınırlı değil")
    if candidate_slot not in local_prompt or "yalnız" not in local_prompt.lower():
        failures.append("local repair prompt'u hedef adayı sınırlamıyor")
    return failures


def _check_generation_batching(cfg) -> list[str]:
    failures = []
    slots = build_plan(cfg)
    groups = _group_slots_by_generator(slots, 3)
    if any(not 1 <= len(group) <= 3 for group in groups):
        failures.append("generation batch boyutu 1–3 sınırını aşıyor")
    if any(len({slot["generator_id"] for slot in group}) != 1 for group in groups):
        failures.append("aynı batch içinde iki generator karıştı")
    grouped_ids = sorted(slot["slot_id"] for group in groups for slot in group)
    if grouped_ids != sorted(slot["slot_id"] for slot in slots):
        failures.append("generation batching slot kaybetti veya çoğalttı")
    if len(groups) != 200:
        failures.append(f"600 family batch=3 ile 200 çağrı olmalı; bulundu: {len(groups)}")
    schema = generation_batch_schema(3)["schema"]["properties"]["families"]
    if schema.get("minItems") != 3 or schema.get("maxItems") != 3:
        failures.append("batch structured-output şeması tam üç family zorlamıyor")
    prompt = build_generation_batch_prompt(slots[:3])
    if "`families`" not in prompt or any(slot["semantic_frame_id"] not in prompt for slot in slots[:3]):
        failures.append("batch prompt family sırasını/frame kimliklerini taşımıyor")

    batch = groups[0]
    calls = []

    class FakeGenerator:
        def call_json(self, system, prompt, schema, task):
            calls.append((prompt, schema, task))
            return SimpleNamespace(
                data={"families": [
                    {"semantic_frame_id": slot["semantic_frame_id"]} for slot in batch
                ]},
                provider="fixture", model="fixture", request_hash="fixture",
                cache_hit=False, usage={}, actual_model="fixture", route_provider="fixture",
            )

    entries = [{
        "slot": slot,
        "prompt_slot": slot,
        "owner": f"owner-{index}",
        "stage_callback": None,
        "start_refill_round": 0,
    } for index, slot in enumerate(batch)]

    def fake_process_slot(*args):
        initial_generation = args[7]
        if initial_generation is None:
            raise AssertionError("batch family _process_slot'a taşınmadı")
        return "accepted", {"slot_id": args[0]["slot_id"]}

    with patch("test.pipeline._process_slot", side_effect=fake_process_slot):
        outcomes = _process_generation_batch(
            entries,
            cfg,
            {batch[0]["generator_id"]: FakeGenerator()},
            {},
            None,
            threading.Lock(),
        )
    if len(calls) != 1 or len(outcomes) != len(batch):
        failures.append("üçlü batch tek provider çağrısından üç family üretmedi")

    def one_family_fails(*args):
        if args[0]["slot_id"] == batch[0]["slot_id"]:
            raise RuntimeError("fixture family error")
        return "accepted", {"slot_id": args[0]["slot_id"]}

    with patch("test.pipeline._process_slot", side_effect=one_family_fails):
        isolated = _process_generation_batch(
            entries,
            cfg,
            {batch[0]["generator_id"]: FakeGenerator()},
            {},
            None,
            threading.Lock(),
        )
    if [row[2] for row in isolated] != ["failed", "accepted", "accepted"]:
        failures.append("tek family istisnası batch kardeşlerine yayıldı")
    return failures


def _check_judge_blindness(cfg) -> list[str]:
    """Semantic judging is feature-blind; both judge stages stay role/position blind."""
    failures = []
    slot = _fixture_slot(cfg)
    family = normalize_family(_fixture_raw(), slot)
    semantic_prompt = build_semantic_judge_prompt(family, "a")
    payload = json.loads(
        semantic_prompt[semantic_prompt.index("{"): semantic_prompt.rindex("}") + 1]
    )
    banned = {"gold_id", "role", "subtype", "relevance", "qrels", "candidate_slot",
              "morph_relation", "hard_profile", "edit_script", "critical_word", "feature_delta"}

    def _keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from _keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from _keys(value)

    for leaked in sorted(banned & set(_keys(payload))):
        failures.append(f"semantic judge payload etiket alanı sızdırıyor: {leaked}")
    if "target_feature" in payload or family["target_feature"] in semantic_prompt:
        failures.append("semantic judge target_feature görüyor")
    if set(payload["candidates"][0]) != {"id", "text"}:
        failures.append(f"judge adayları yalnız id+text olmalı: {sorted(payload['candidates'][0])}")
    if family["gold_id"] in json.dumps(payload, ensure_ascii=False).replace(
        f"\"{family['gold_id']}\"", "", 1
    ).replace(family["family_id"], ""):
        failures.append("judge payload gold_id'yi işaretli biçimde taşıyor")

    # Position/ID blindness: over many families the gold must not settle on one slot or one
    # `_cNN` suffix, or the judge could answer from ordering alone.
    slots, ids = Counter(), Counter()
    for index in range(120):
        probe_slot = make_slot(index, cfg)
        probe = normalize_family(_fixture_raw(), probe_slot)
        order = [c["id"] for c in _blind_candidates(probe, "semantic_a")]
        slots[order.index(probe["gold_id"])] += 1
        ids[probe["gold_id"].rsplit("_c", 1)[-1]] += 1
    if len(slots) < 11 or max(slots.values()) > 40:
        failures.append(f"judge aday sırasında gold konumu dengesiz: {dict(sorted(slots.items()))}")
    if len(ids) < 11 or max(ids.values()) > 40:
        failures.append(f"gold candidate ID soneki dengesiz: {dict(sorted(ids.items()))}")
    order_a = [c["id"] for c in _blind_candidates(family, "semantic_a")]
    order_b = [c["id"] for c in _blind_candidates(family, "semantic_b")]
    if order_a == order_b:
        failures.append("semantic judge counterbalanced candidate sıraları aynı")
    morph_prompt = build_morphology_judge_prompt(family)
    morph_payload = json.loads(morph_prompt[morph_prompt.index("{"): morph_prompt.rindex("}") + 1])
    if morph_payload.get("target_feature") != family["target_feature"]:
        failures.append("morphology judge target_feature görmüyor")
    for leaked in sorted(banned & set(_keys(morph_payload))):
        failures.append(f"morphology judge payload etiket alanı sızdırıyor: {leaked}")
    return failures


def _check_model_family_gates() -> list[str]:
    """Distinct-family and forbidden-family enforcement, including the route-prefix aliases that
    a literal `provider/` comparison used to wave through."""
    failures = []
    for model_id, expected in (
        ("google/gemini-2.5-pro", "google"),
        ("google-vertex/gemini-2.5-pro", "google"),
        ("gemini-2.5-pro", "google"),
        ("openai/gpt-5", "openai"),
        ("gpt-5.6-sol", "openai"),
        ("anthropic/claude-opus-4", "anthropic"),
        ("claude-sonnet-test", "anthropic"),
    ):
        if model_family(model_id) != expected:
            failures.append(f"model_family({model_id}) -> {model_family(model_id)}, beklenen {expected}")

    base = json.loads((Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8"))

    def _rejects(model_a, model_b, semantic_model, morphology_model) -> bool:
        cfg = deepcopy(base)
        cfg["generation"]["generators"][0]["model"] = model_a
        cfg["generation"]["generators"][1]["model"] = model_b
        cfg["generation"]["judges"]["semantic"]["model"] = semantic_model
        cfg["generation"]["judges"]["morphology"]["model"] = morphology_model
        os.environ.setdefault("OPENROUTER_API_KEY", "selftest-placeholder")
        try:
            validate_config(cfg, runtime=True)
            return False
        except ConfigError:
            return True

    for label, models, want_reject in (
        ("yasak google ailesi", ("google/gemini-2.5-pro", "openai/gpt-5", "qwen/qwen3", "mistralai/small"), True),
        ("google-vertex takma adı", ("google-vertex/gemini-2.5-pro", "openai/gpt-5", "qwen/qwen3", "mistralai/small"), True),
        ("öneksiz gemini kimliği", ("gemini-2.5-pro", "openai/gpt-5", "qwen/qwen3", "mistralai/small"), True),
        ("aynı aileden iki generator", ("openai/gpt-5", "openai/gpt-5-mini", "qwen/qwen3", "mistralai/small"), True),
        ("iki google rotası tek aile sayılmalı",
         ("google/gemini-2.5-pro", "google-vertex/gemini-2.5-pro", "qwen/qwen3", "mistralai/small"), True),
        ("judge generator ailesini tekrar ediyor",
         ("openai/gpt-5", "anthropic/claude-opus-4", "openai/gpt-5-mini", "mistralai/small"), True),
        ("geçerli dört ayrı aile",
         ("openai/gpt-5", "anthropic/claude-opus-4", "qwen/qwen3", "mistralai/small"), False),
    ):
        if _rejects(*models) is not want_reject:
            verb = "reddetmedi" if want_reject else "gereksiz yere reddetti"
            failures.append(f"model aile kapısı {label} durumunu {verb}")
    return failures


def _check_claude_cli_adapter() -> list[str]:
    failures = []
    with TemporaryDirectory() as temporary:
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            return SimpleNamespace(
                returncode=0, stderr="", stdout=json.dumps({
                    "type": "result", "subtype": "success", "is_error": False,
                    "structured_output": {"ok": True},
                    "modelUsage": {"claude-fixture-model": {"inputTokens": 10}},
                    "num_turns": 1,
                }),
            )

        provider = ClaudeCliProvider(
            {
                "provider": "claude_cli", "model": "claude-fixture-model",
                "executable": "/mock/claude", "workdir": temporary,
            },
            Path(temporary) / "cache", {"selftest": True},
        )
        schema = {
            "name": "fixture", "schema": {
                "type": "object", "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"], "additionalProperties": False,
            },
        }
        with patch("test.providers.subprocess.run", side_effect=fake_run):
            response = provider.call_json("system", "prompt", schema, "selftest")
        command = captured.get("command", [])
        if response.data != {"ok": True} or response.actual_model != "claude-fixture-model":
            failures.append("Claude CLI structured output/provenance ayrıştırması yanlış")
        for flag in ("--model", "--json-schema", "--tools", "--no-session-persistence"):
            if flag not in command:
                failures.append(f"Claude CLI güvenli structured-output flag'i eksik: {flag}")
    return failures


if __name__ == "__main__":
    errors = run()
    if errors:
        for error in errors:
            print("FAIL:", error)
        raise SystemExit(1)
    print("OK: test pipeline self-test")
