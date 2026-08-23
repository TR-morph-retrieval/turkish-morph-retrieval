"""Deterministic family/corpus gates and blind-judge interpretation."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any

from .dataset_memory import family_memory_tags, semantic_profile_problems

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9]+", re.I)
_META_PATTERNS = ("kaydı bul", "kaydi bul", "arıyorum", "ariyorum", "hangi kayıt", "hangi kayit")
_ABBREVIATIONS = ("Dr.", "Prof.", "Doç.", "Sn.", "vb.", "vs.", "örn.", "T.C.")
_CONTENT_STOPWORDS = {
    "acaba", "ama", "ancak", "artık", "bile", "bir", "biz", "bu", "bunu", "bunun",
    "da", "daha", "de", "diye", "en", "fakat", "gibi", "hem", "her", "için", "ile",
    "ise", "ki", "mi", "mı", "mu", "mü", "ne", "o", "olarak", "sonra", "şu", "ve",
    "veya", "ya", "yalnız",
}


def tr_lower(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(tr_lower(text))


def sentence_count(text: str) -> int:
    clean = text.strip()
    for abbreviation in _ABBREVIATIONS:
        clean = clean.replace(abbreviation, abbreviation.replace(".", "∯"))
    return len(re.findall(r"[.!?]+(?:[\"'”’)]*)?(?=\s|$)", clean))


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = " ".join(tokens(text))
    return {normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1))}


def jaccard(left: set, right: set) -> float:
    return len(left & right) / max(1, len(left | right))


def _contains_word(text: str, word: str) -> bool:
    return tr_lower(word).strip(".,;:!?()[]{}\"'") in tr_lower(text)


def _critical_skeleton(sentence: str, critical_word: str) -> list[str] | None:
    """Replace one token sequence with a marker for strict minimal-pair comparison."""
    sentence_tokens = tokens(sentence)
    word_tokens = tokens(critical_word)
    if not word_tokens:
        return None
    for start in range(len(sentence_tokens) - len(word_tokens) + 1):
        if sentence_tokens[start : start + len(word_tokens)] == word_tokens:
            return sentence_tokens[:start] + ["<morph>"] + sentence_tokens[start + len(word_tokens) :]
    return None


def _word_overlap(query: str, text: str) -> float:
    query_terms = set(tokens(query))
    return len(query_terms & set(tokens(text))) / max(1, len(query_terms))


def _word_jaccard(left: str, right: str) -> float:
    return jaccard(set(tokens(left)), set(tokens(right)))


def _token_edit_distance(left: str, right: str) -> int:
    """Small dependency-free Levenshtein distance over word tokens."""
    a, b = tokens(left), tokens(right)
    previous = list(range(len(b) + 1))
    for i, left_token in enumerate(a, start=1):
        current = [i]
        for j, right_token in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def _token_sequence_ratio(left: str, right: str) -> float:
    return SequenceMatcher(a=tokens(left), b=tokens(right), autojunk=False).ratio()


_HARD_RELATIONS = {
    "minimal_morph_negative": "feature_changed",
    "controlled_morph_negative": "feature_changed",
    "same_lemma_wrong_inflection": "wrong_inflection",
    "related_feature_negative": "feature_changed",
    "same_morph_wrong_content": "same_feature_wrong_content",
    "state_participant_time_trap": "target_preserved",
    "close_paraphrase_wrong_meaning": "target_preserved",
    "argument_role_reversal": "target_preserved",
    "scope_attachment_trap": "scope_changed",
    "morph_distractor": "target_preserved",
    "partial_chain_negative": "chain_partial",
    "allomorph_form_function_trap": "feature_changed",
    "noun_possessor_number_trap": "feature_changed",
    "lexical_retrieval_trap": "target_preserved",
    "semantic_retrieval_hard": "target_preserved",
}


def _inject_candidate_metadata(candidates: list[dict], slot: dict) -> None:
    """Derive labels from the trusted plan instead of asking the LLM to echo them."""
    hard_types = {row["slot"]: row["subtype"] for row in slot["hard_profile"]}
    for candidate in candidates:
        candidate_slot = candidate.get("candidate_slot")
        if candidate_slot == "positive_01":
            candidate.update({
                "role": "positive",
                "subtype": "equivalence_positive",
                "morph_relation": (
                    "allomorph_equivalent"
                    if slot["objective"] == "allomorph_invariance"
                    else "target_preserved"
                ),
            })
        elif candidate_slot in {"easy_01", "easy_02"}:
            candidate.update({
                "role": "easy_negative",
                "subtype": "easy_negative",
                "morph_relation": "same_domain_off_intent",
            })
        elif candidate_slot in hard_types:
            subtype = hard_types[candidate_slot]
            candidate.update({
                "role": "hard_negative",
                "subtype": subtype,
                "morph_relation": _HARD_RELATIONS[subtype],
            })


def _content_prefixes(text: str) -> set[str]:
    values = {
        token[:5] for token in tokens(text)
        if len(token) >= 4 and token not in _CONTENT_STOPWORDS
    }
    return values or {token[:5] for token in tokens(text) if len(token) >= 3}


def _critical_query_sentence(family: dict[str, Any]) -> str:
    query = str(family.get("query", ""))
    critical = str(family.get("critical_word_query", ""))
    for sentence in re.split(r"(?<=[.!?])\s+", query):
        if critical and _contains_word(sentence, critical):
            return sentence
    return query


def lexical_family_audit(family: dict[str, Any]) -> dict[str, Any]:
    """Measure lexical balance on critical sentences, where candidate meaning differs."""
    candidates = family.get("candidates", [])
    positive = next((row for row in candidates if row.get("role") == "positive"), None)
    hards = [row for row in candidates if row.get("role") == "hard_negative"]
    if positive is None or not hards:
        return {}
    query = _critical_query_sentence(family)
    gold_overlap = _word_overlap(query, str(positive.get("critical_sentence", "")))
    hard_overlaps = [
        _word_overlap(query, str(candidate.get("critical_sentence", "")))
        for candidate in hards
    ]
    ordered = sorted(hard_overlaps)
    midpoint = len(ordered) // 2
    hard_median = (
        ordered[midpoint] if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    query_content = _content_prefixes(query)
    content_recalls = [
        len(query_content & _content_prefixes(str(candidate.get("critical_sentence", ""))))
        / max(1, len(query_content))
        for candidate in hards
    ]
    texts = [str(candidate.get("text", "")) for candidate in candidates]
    gold_index = candidates.index(positive)
    full_word_scores = [_word_overlap(str(family.get("query", "")), text) for text in texts]
    query_grams = char_ngrams(str(family.get("query", "")))
    char_scores = [jaccard(query_grams, char_ngrams(text)) for text in texts]
    bm25_scores = _bm25_scores(str(family.get("query", "")), texts)
    return {
        "gold_word_overlap": round(gold_overlap, 6),
        "hard_word_overlap_median": round(hard_median, 6),
        "gold_hard_median_gap": round(abs(gold_overlap - hard_median), 6),
        "hards_at_or_above_gold": sum(score >= gold_overlap for score in hard_overlaps),
        "hard_query_content_recalls": [round(score, 6) for score in content_recalls],
        "tie_aware_gold_top_probability": {
            "word_overlap": round(_tie_aware_top_probability(full_word_scores, gold_index), 6),
            "character_3gram": round(_tie_aware_top_probability(char_scores, gold_index), 6),
            "bm25": round(_tie_aware_top_probability(bm25_scores, gold_index), 6),
        },
    }


def normalize_family(raw: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    """Assemble controlled passages, then inject trusted metadata and neutral IDs."""
    if not isinstance(raw, dict):
        raise ValueError("Generator çıktısı object olmalı")
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates listesi yok")
    context_sentences = raw.get("context_sentences", [])
    if not isinstance(context_sentences, list):
        raise ValueError("context_sentences liste olmalı")
    candidates = [dict(candidate) for candidate in raw_candidates]
    _inject_candidate_metadata(candidates, slot)
    insert_at = int(slot["critical_sentence_position"]) - 1
    for candidate in candidates:
        sentences = [str(sentence).strip() for sentence in context_sentences]
        sentences.insert(insert_at, str(candidate.get("critical_sentence", "")).strip())
        candidate["text"] = " ".join(sentence for sentence in sentences if sentence)
    shuffle_seed = int(hashlib.sha256(slot["slot_id"].encode()).hexdigest()[:16], 16)
    random.Random(shuffle_seed).shuffle(candidates)
    family_id = f"family_{slot['slot_id']}"
    for index, candidate in enumerate(candidates, start=1):
        candidate["id"] = f"{family_id}_c{index:02d}"
        candidate["relevance"] = int(candidate.get("role") == "positive")

    positives = [candidate for candidate in candidates if candidate.get("role") == "positive"]
    minimal = next(
        (candidate for candidate in candidates if candidate.get("candidate_slot") == "hard_01"),
        None,
    )
    positive_word = positives[0].get("critical_word", "") if len(positives) == 1 else ""
    minimal_word = minimal.get("critical_word", "") if minimal else ""
    strict = bool(slot["strict_minimal_pair"])
    gold_id = positives[0]["id"] if len(positives) == 1 else None
    family = {
        "family_id": family_id,
        "slot_id": slot["slot_id"],
        "target_split": slot["target_split"],
        "generalization_bucket": slot["generalization_bucket"],
        "generalization_tags": list(slot["generalization_tags"]),
        "objective": slot["objective"],
        "macro_phenomenon": slot["macro_phenomenon"],
        "target_feature": slot["feature"]["key"],
        "target_feature_label": slot["feature"]["label"],
        "surface_forms": list(slot["feature"]["surface_forms"]),
        "layer": slot["layer"],
        "domain": slot["domain"],
        "register": slot["register"],
        "query_sentence_count": slot["query_sentence_count"],
        "passage_sentence_count": slot["passage_sentence_count"],
        "critical_sentence_position": slot["critical_sentence_position"],
        "family_mode": slot["family_mode"],
        "query_expression": slot["query_expression"],
        "query_gold_lexical_band": slot["query_gold_lexical_band"],
        "hard_profile": list(slot["hard_profile"]),
        "strict_minimal_pair": bool(slot["strict_minimal_pair"]),
        "generator_id": slot["generator_id"],
        "semantic_frame_id": raw.get("semantic_frame_id"),
        "semantic_profile": raw.get("semantic_profile"),
        "template_id": slot["template"]["id"],
        "critical_lemma": raw.get("critical_lemma"),
        "critical_word_query": raw.get("critical_word_query"),
        "critical_word_positive": positive_word,
        "feature_delta": slot["feature"]["meaning_contrast"],
        "edit_script": {
            "applies": strict,
            "positive_form": positive_word if strict else "",
            "minimal_negative_form": minimal_word if strict else "",
            "operation": f"yalnız {slot['feature']['key']} biçimini değiştir" if strict else "",
            "changed_feature": slot["feature"]["key"] if strict else "",
            "invariants": ["lemma", "token_order", "event"] if strict else [],
        },
        "query": raw.get("query"),
        "context_sentences": context_sentences,
        "candidates": candidates,
        "gold_id": gold_id,
        "qrels": {candidate["id"]: candidate["relevance"] for candidate in candidates},
        "source_type": "llm_generated_pending_cascade_judge",
    }
    family["memory_tags"] = family_memory_tags(family)
    return family


def validate_family(family: dict[str, Any], slot: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    query = family.get("query")
    candidates = family.get("candidates")
    if not isinstance(query, str) or not query.strip():
        problems.append("query boş veya string değil")
        return problems
    if not isinstance(candidates, list):
        problems.append("candidates listesi yok")
        return problems
    if len(candidates) != 11:
        problems.append(f"11 aday gerekli; bulunan {len(candidates)}")

    roles = Counter(candidate.get("role") for candidate in candidates)
    expected_roles = cfg["candidate_counts"]
    if dict(roles) != expected_roles:
        problems.append(f"rol sayıları yanlış: {dict(roles)} != {expected_roles}")

    hard_expected = {
        item["slot"]: item["subtype"] for item in slot["hard_profile"]
    }
    hard_actual = {
        candidate.get("candidate_slot"): candidate.get("subtype")
        for candidate in candidates if candidate.get("role") == "hard_negative"
    }
    if len(hard_actual) != 8 or hard_actual != hard_expected:
        problems.append(f"uyarlanabilir sekiz hard slot yanlış: {hard_actual} != {hard_expected}")
    positives = [candidate for candidate in candidates if candidate.get("role") == "positive"]
    if len(positives) == 1 and positives[0].get("subtype") != "equivalence_positive":
        problems.append("positive subtype equivalence_positive olmalı")
    if len(positives) == 1 and positives[0].get("candidate_slot") != "positive_01":
        problems.append("positive candidate_slot positive_01 olmalı")
    easies = [candidate for candidate in candidates if candidate.get("role") == "easy_negative"]
    if any(candidate.get("subtype") != "easy_negative" for candidate in easies):
        problems.append("iki easy adayın subtype'ı easy_negative olmalı")
    if {candidate.get("candidate_slot") for candidate in easies} != {"easy_01", "easy_02"}:
        problems.append("easy candidate_slot değerleri easy_01/easy_02 olmalı")
    expected_slots = {"positive_01", "easy_01", "easy_02", *hard_expected}
    actual_slots = [candidate.get("candidate_slot") for candidate in candidates]
    if len(set(actual_slots)) != 11 or set(actual_slots) != expected_slots:
        problems.append(f"candidate_slot kapsamı yanlış: {actual_slots}")

    ids = [candidate.get("id") for candidate in candidates]
    if len(ids) != len(set(ids)) or any(not candidate_id for candidate_id in ids):
        problems.append("candidate id'leri eksik veya benzersiz değil")
    expected_qrels = {
        candidate.get("id"): int(candidate.get("role") == "positive") for candidate in candidates
    }
    if family.get("qrels") != expected_qrels:
        problems.append("family qrels 1 gold + 10 negative ile uyuşmuyor")
    if any(candidate.get("relevance") != expected_qrels.get(candidate.get("id")) for candidate in candidates):
        problems.append("candidate relevance etiketi role ile uyuşmuyor")
    texts = [candidate.get("text") for candidate in candidates]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        problems.append("boş/non-string candidate text var")
    elif len({tr_lower(text).strip() for text in texts}) != len(texts):
        problems.append("candidate metinleri arasında birebir tekrar var")

    wanted_query_sentences = int(family["query_sentence_count"])
    wanted_passage_sentences = int(family["passage_sentence_count"])
    if sentence_count(query) != wanted_query_sentences:
        problems.append(
            f"query {wanted_query_sentences} yerine {sentence_count(query)} cümle"
        )
    context_sentences = family.get("context_sentences")
    if not isinstance(context_sentences, list):
        problems.append("context_sentences liste değil")
        context_sentences = []
    if len(context_sentences) != wanted_passage_sentences - 1:
        problems.append(
            f"ortak bağlam {wanted_passage_sentences - 1} yerine {len(context_sentences)} cümle"
        )
    for index, context in enumerate(context_sentences, start=1):
        if not isinstance(context, str) or sentence_count(context) != 1:
            problems.append(f"context_sentences[{index}] tek tam cümle değil")
    for candidate in candidates:
        text = candidate.get("text", "")
        critical_sentence = candidate.get("critical_sentence", "")
        if not isinstance(critical_sentence, str) or sentence_count(critical_sentence) != 1:
            problems.append(f"{candidate.get('id')} critical_sentence tek tam cümle değil")
        if isinstance(text, str) and sentence_count(text) != wanted_passage_sentences:
            problems.append(
                f"{candidate.get('id')} {wanted_passage_sentences} yerine "
                f"{sentence_count(text)} cümle"
            )

    is_focus_question = family.get("target_feature") == "Q.PART.SCOPE"
    if is_focus_question and not query.rstrip().endswith("?"):
        problems.append("Q.PART.SCOPE query doğal bir odak sorusu olmalı")
    elif not is_focus_question and query.rstrip().endswith("?"):
        problems.append("query soru cümlesi olamaz")
    lowered_query = tr_lower(query)
    if any(pattern in lowered_query for pattern in _META_PATTERNS):
        problems.append("query meta-arama dili içeriyor")

    if family.get("semantic_frame_id") != slot["semantic_frame_id"]:
        problems.append("semantic_frame_id SLOT ile uyuşmuyor")
    # New provider-facing generation schema requires this profile. Historical curated previews
    # predate dataset memory, so they remain reproducible when the field is absent; if present it
    # is always validated.
    if family.get("semantic_profile") is not None:
        problems.extend(semantic_profile_problems(family.get("semantic_profile")))
    if family.get("template_id") != slot["template"]["id"]:
        problems.append("template_id SLOT ile uyuşmuyor")
    for field in ("family_mode", "query_expression", "query_gold_lexical_band"):
        if family.get(field) != slot[field]:
            problems.append(f"{field} SLOT ile uyuşmuyor")
    if bool(family.get("strict_minimal_pair")) != (slot["family_mode"] == "strict_minimal"):
        problems.append("strict_minimal_pair family_mode ile uyuşmuyor")
    for name in ("critical_lemma", "critical_word_query", "critical_word_positive", "feature_delta"):
        if not isinstance(family.get(name), str) or not family[name].strip():
            problems.append(f"{name} eksik")
    if isinstance(family.get("critical_word_query"), str) and not _contains_word(query, family["critical_word_query"]):
        problems.append("critical_word_query query içinde yok")
    if positives and isinstance(family.get("critical_word_positive"), str):
        if not _contains_word(positives[0].get("text", ""), family["critical_word_positive"]):
            problems.append("critical_word_positive positive metninde yok")

    query_gold_edits = None
    query_gold_sequence_ratio = None
    query_gold_word_jaccard = None
    if positives:
        query_critical = _critical_query_sentence(family)
        gold_critical = positives[0].get("critical_sentence", "")
        query_gold_edits = _token_edit_distance(query_critical, gold_critical)
        query_gold_sequence_ratio = _token_sequence_ratio(query_critical, gold_critical)
        query_gold_word_jaccard = _word_jaccard(query_critical, gold_critical)
        minimum_edits = int(cfg["quality"].get("query_gold_token_edits_min", 2))
        maximum_sequence = float(
            cfg["quality"].get("query_gold_token_sequence_ratio_max", 0.82)
        )
        if query_gold_edits < minimum_edits or query_gold_sequence_ratio > maximum_sequence:
            problems.append(
                "query ile gold tek-kelime/yakın-kopya: "
                f"token-edits={query_gold_edits}, sequence-ratio={query_gold_sequence_ratio:.3f}"
            )
        band = cfg["quality"]["query_gold_lexical_bands"][slot["query_gold_lexical_band"]]
        query_gold_lexical_band_ok = (
            float(band["min"]) <= query_gold_word_jaccard <= float(band["max"])
        )

    for candidate in candidates:
        critical = candidate.get("critical_word")
        if not isinstance(critical, str) or not critical.strip():
            problems.append(f"{candidate.get('id')} critical_word eksik")
        elif not _contains_word(candidate.get("critical_sentence", ""), critical):
            problems.append(f"{candidate.get('id')} critical_word kritik cümlede yok")

    edit_script = family.get("edit_script")
    if not isinstance(edit_script, dict):
        problems.append("edit_script eksik veya object değil")
    elif bool(edit_script.get("applies")) != bool(slot["strict_minimal_pair"]):
        problems.append("edit_script.applies strict_minimal_pair planıyla uyuşmuyor")
    if slot["strict_minimal_pair"] and positives:
        minimal = next(
            (candidate for candidate in candidates if candidate.get("candidate_slot") == "hard_01"),
            None,
        )
        if minimal is None:
            problems.append("strict minimal pair için hard_01 yok")
        else:
            positive_skeleton = _critical_skeleton(
                positives[0].get("critical_sentence", ""), positives[0].get("critical_word", "")
            )
            negative_skeleton = _critical_skeleton(
                minimal.get("critical_sentence", ""), minimal.get("critical_word", "")
            )
            if positive_skeleton is None or negative_skeleton is None:
                problems.append("strict minimal pair kritik sözcüğü ayrıştırılamadı")
            elif positive_skeleton != negative_skeleton:
                problems.append("strict minimal pair positive/hard_01 yalnız kritik biçimde ayrışmıyor")
            if minimal.get("subtype") != "minimal_morph_negative":
                problems.append("strict minimal pair hard_01 minimal_morph_negative olmalı")
            if isinstance(edit_script, dict):
                expected_forms = {
                    tr_lower(positives[0].get("critical_word", "")).strip(),
                    tr_lower(minimal.get("critical_word", "")).strip(),
                }
                recorded_forms = {
                    tr_lower(str(edit_script.get("positive_form", ""))).strip(),
                    tr_lower(str(edit_script.get("minimal_negative_form", ""))).strip(),
                }
                if expected_forms != recorded_forms:
                    problems.append("edit_script yüzey biçimleri positive/hard_01 ile uyuşmuyor")
                if not str(edit_script.get("changed_feature", "")).strip():
                    problems.append("strict edit_script.changed_feature boş")

    if family["objective"] == "allomorph_invariance":
        if positives and positives[0].get("morph_relation") != "allomorph_equivalent":
            problems.append("allomorph family positive ilişkisi allomorph_equivalent olmalı")
        bad = [candidate.get("id") for candidate in candidates if candidate.get("role") != "positive" and candidate.get("morph_relation") == "allomorph_equivalent"]
        if bad:
            problems.append(f"geçerli allomorph negatif yapılmış: {bad}")
    elif positives and positives[0].get("morph_relation") != "target_preserved":
        problems.append("semantic/composition positive ilişkisi target_preserved olmalı")

    lengths = [len(tokens(text)) for text in texts if isinstance(text, str)]
    if len(lengths) == 11 and min(lengths) > 0:
        ratio = max(lengths) / min(lengths)
        if ratio > float(cfg["quality"]["candidate_token_ratio_max"]):
            problems.append(f"candidate token uzunluk oranı yüksek: {ratio:.2f}")
        if positives:
            gold_len = len(tokens(positives[0]["text"]))
            median = sorted(lengths)[len(lengths) // 2]
            if gold_len / max(1, median) > float(cfg["quality"]["gold_to_median_token_ratio_max"]):
                problems.append(f"gold uzunluk bias'ı: gold/median={gold_len / max(1, median):.2f}")

    lexical = lexical_family_audit(family)
    if query_gold_edits is not None:
        lexical["query_gold_token_edits"] = query_gold_edits
        lexical["query_gold_token_sequence_ratio"] = round(query_gold_sequence_ratio, 6)
        lexical["query_gold_word_jaccard"] = round(query_gold_word_jaccard, 6)
        lexical["query_gold_lexical_band_ok"] = query_gold_lexical_band_ok
    family["lexical_audit"] = lexical
    if lexical:
        quality = cfg["quality"]
        mode_gates = quality["family_mode_lexical_gates"][slot["family_mode"]]
        minimum_close_hards = int(mode_gates["hards_at_or_above_gold_min"])
        if lexical["hards_at_or_above_gold"] < minimum_close_hards:
            problems.append(
                "lexical overlap gold'u ele veriyor: "
                f"gold kadar örtüşen hard={lexical['hards_at_or_above_gold']} < {minimum_close_hards}"
            )
        max_gap = float(mode_gates["gold_hard_median_gap_max"])
        if lexical["gold_hard_median_gap"] > max_gap:
            problems.append(
                "gold-hard lexical overlap dengesiz: "
                f"median fark={lexical['gold_hard_median_gap']:.3f} > {max_gap:.3f}"
            )
        content_threshold = float(quality["hard_query_content_recall_min"])
        content_count = sum(
            score >= content_threshold for score in lexical["hard_query_content_recalls"]
        )
        content_minimum = int(mode_gates["content_preserving_hards_min"])
        if slot["query_expression"] == "semantic_paraphrase" or slot["query_gold_lexical_band"] == "low":
            content_minimum = 0
        lexical["content_preserving_hards"] = content_count
        if content_count < content_minimum:
            problems.append(
                "aynı içerik sözcüklerini koruyan hard sayısı düşük: "
                f"{content_count} < {content_minimum}"
            )

    return sorted(set(problems))


def interpret_semantic_judges(
    family: dict[str, Any], verdicts: list[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Interpret one or more feature-blind semantic passes."""
    problems: list[str] = []
    candidates = {candidate["id"]: candidate for candidate in family["candidates"]}
    relevance_floor = int(cfg["quality"]["semantic_relevance_confidence_min"])
    quality_floor = int(cfg["quality"]["semantic_quality_confidence_min"])
    summaries = []
    decision_signatures = []

    for index, verdict in enumerate(verdicts, start=1):
        label = f"semantic_{index}"
        def id_list(name: str) -> list[str]:
            value = verdict.get(name, [])
            return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else []

        answers = id_list("fully_relevant_candidate_ids")
        unnatural = id_list("unnatural_candidate_ids")
        inconsistent = id_list("internally_inconsistent_candidate_ids")
        unknown_ids = sorted((set(answers) | set(unnatural) | set(inconsistent)) - set(candidates))
        support_map = {candidate_id: candidate_id in answers for candidate_id in candidates}
        support_mismatches = [
            candidate_id for candidate_id, supports in support_map.items()
            if supports != (candidate_id == family["gold_id"])
        ]
        family_naturalness = verdict.get("family_naturalness", 0)
        confidence = verdict.get("confidence", 0)
        relevance_authoritative = (
            isinstance(confidence, int) and confidence >= relevance_floor
            and not verdict.get("abstain")
        )
        quality_authoritative = (
            isinstance(confidence, int) and confidence >= quality_floor
            and not verdict.get("abstain")
        )
        if relevance_authoritative:
            if unknown_ids:
                problems.append(f"{label} bilinmeyen candidate ID verdi: {unknown_ids}")
            if support_mismatches:
                problems.append(f"{label} query desteği uyuşmazlığı: {support_mismatches}")
        if quality_authoritative:
            if inconsistent:
                problems.append(f"{label} iç tutarsız aday buldu: {inconsistent}")
            if unnatural:
                problems.append(f"{label} doğal olmayan aday buldu: {unnatural}")
            naturalness_min = int(cfg["quality"]["judge_naturalness_min"])
            if not isinstance(family_naturalness, int) or family_naturalness < naturalness_min:
                problems.append(
                    f"{label} family naturalness {family_naturalness}/{naturalness_min}"
                )
            if verdict.get("length_or_style_artifact"):
                problems.append(f"{label} uzunluk/üslup artefaktı buldu")

        decision_signatures.append((frozenset(answers), support_map))
        summaries.append({
            "answers_query": answers,
            "confidence": confidence,
            "abstain": bool(verdict.get("abstain")),
            "family_naturalness": family_naturalness,
            "candidate_naturalness_min": 1 if unnatural else 5,
            "support_mismatches": support_mismatches,
            "unnatural_candidate_ids": unnatural,
            "internally_inconsistent_candidates": inconsistent,
            "length_or_style_artifact": bool(verdict.get("length_or_style_artifact")),
            "authoritative": relevance_authoritative or quality_authoritative,
            "relevance_authoritative": relevance_authoritative,
            "quality_authoritative": quality_authoritative,
            "notes": verdict.get("notes", ""),
        })

    order_stable = len(decision_signatures) < 2 or all(
        signature == decision_signatures[0] for signature in decision_signatures[1:]
    )
    if len(decision_signatures) >= 2 and not order_stable:
        problems.append("semantic judge candidate sırası değişince karar değiştirdi")
    return sorted(set(problems)), {
        "order_stable": order_stable,
        "passes": summaries,
        "family_naturalness": min(
            (row["family_naturalness"] for row in summaries), default=0
        ),
        "candidate_naturalness_min": min(
            (row["candidate_naturalness_min"] for row in summaries), default=0
        ),
        "confidence_min": min((row["confidence"] for row in summaries), default=0),
    }


def interpret_morphology_judge(
    family: dict[str, Any], verdict: dict[str, Any], cfg: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Interpret the independent feature-aware morphology pass without subtype matching."""
    problems: list[str] = []
    candidates = {candidate["id"]: candidate for candidate in family["candidates"]}
    def id_list(name: str) -> list[str]:
        value = verdict.get(name, [])
        return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else []

    target_matches = id_list("target_matching_candidate_ids")
    morphology_failures = id_list("morphologically_invalid_candidate_ids")
    unclear_ids = id_list("unclear_candidate_ids")
    unknown_ids = sorted(
        (set(target_matches) | set(morphology_failures) | set(unclear_ids)) - set(candidates)
    )
    gold_target_status = "matches_target" if family["gold_id"] in target_matches else "not_confirmed"
    confidence = verdict.get("confidence", 0)
    floor = int(cfg["quality"]["morphology_judge_confidence_min"])
    authoritative = (
        isinstance(confidence, int) and confidence >= floor and not verdict.get("abstain")
    )
    if authoritative:
        if unknown_ids:
            problems.append(f"morphology judge bilinmeyen candidate ID verdi: {unknown_ids}")
        if morphology_failures:
            problems.append(f"morphology judge bozuk biçimbilim işaretledi: {morphology_failures}")
        if gold_target_status != "matches_target":
            problems.append(f"morphology judge gold hedef özelliğini doğrulamadı: {gold_target_status}")
        if verdict.get("allomorph_treated_as_wrong"):
            problems.append("morphology judge geçerli allomorfu yanlış saydı")
    return sorted(set(problems)), {
        "confidence": confidence,
        "abstain": bool(verdict.get("abstain")),
        "authoritative": authoritative,
        "morphology_failures": morphology_failures,
        "unclear_candidate_ids": unclear_ids,
        "gold_target_status": gold_target_status,
        "target_matching_candidate_ids": target_matches,
        "allomorph_treated_as_wrong": bool(verdict.get("allomorph_treated_as_wrong")),
        "notes": verdict.get("notes", ""),
    }


def quality_score(family: dict[str, Any]) -> float:
    judging = family.get("qc", {}).get("judging", {})
    semantic = judging.get("semantic", family.get("qc", {}).get("judge", {}))
    morphology = judging.get("morphology", {})
    lexical = family.get("lexical_audit", {})
    lengths = [len(tokens(candidate["text"])) for candidate in family["candidates"]]
    balance = min(lengths) / max(lengths) if lengths and max(lengths) else 0.0
    lexical_balance = 1.0 - min(1.0, float(lexical.get("gold_hard_median_gap", 1.0)))
    close_hard_share = min(1.0, float(lexical.get("hards_at_or_above_gold", 0)) / 8.0)
    cheap_top = lexical.get("tie_aware_gold_top_probability", {})
    artefact_resistance = (
        1.0 - sum(float(value) for value in cheap_top.values()) / len(cheap_top)
        if cheap_top else 0.0
    )
    return (
        float(semantic.get("family_naturalness", 0))
        + float(bool(semantic.get("order_stable", False)))
        + float(morphology.get("confidence", 0)) / 100.0
        + balance
        + lexical_balance
        + 0.5 * close_hard_share
        + artefact_resistance
    )


def corpus_problems(items: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    ids = [item["family_id"] for item in items]
    if len(ids) != len(set(ids)):
        problems.append("family_id tekrarı var")
    frames = [str(item.get("semantic_frame_id", "")).strip() for item in items]
    repeated_frames = sorted(
        frame for frame, count in Counter(frames).items() if frame and count > 1
    )
    if repeated_frames:
        problems.append(f"semantic_frame_id tekrarı var: {repeated_frames[:20]}")
    normalized_queries: dict[str, list[str]] = defaultdict(list)
    normalized_candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        normalized_queries[" ".join(tokens(item["query"]))].append(item["family_id"])
        for candidate in item["candidates"]:
            normalized_candidates[" ".join(tokens(candidate["text"]))].append(
                (item["family_id"], candidate["id"])
            )
    for text, family_ids in normalized_queries.items():
        if text and len(set(family_ids)) > 1:
            problems.append(f"birebir query kopyası: {sorted(set(family_ids))}")
    for text, rows in normalized_candidates.items():
        families = {family_id for family_id, _ in rows}
        if text and len(families) > 1:
            problems.append(f"cross-family birebir candidate kopyası: {rows[:8]}")

    # Easy distractors başka bir family'nin cevabı olarak tekrar kullanılamaz. Genel duplicate
    # kapısı da bunu yakalar; rol-duyarlı kapı daha düşük eşikle ve daha açık hata verir.
    gold_easy_records = []
    for item in items:
        for candidate in item["candidates"]:
            role = candidate.get("role")
            if role in {"positive", "easy_negative"}:
                gold_easy_records.append(
                    (item["family_id"], candidate["id"], role, candidate["text"])
                )
    gold_easy_exact: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for family_id, candidate_id, role, text in gold_easy_records:
        gold_easy_exact[" ".join(tokens(text))].append((family_id, candidate_id, role))
    for text, rows in gold_easy_exact.items():
        roles = {row[2] for row in rows}
        families = {row[0] for row in rows}
        if text and len(families) > 1 and roles == {"positive", "easy_negative"}:
            problems.append(f"başka family gold'u easy olarak kullanılmış: {rows[:8]}")

    query_grams = [(item["family_id"], char_ngrams(item["query"])) for item in items]
    threshold = float(cfg["quality"]["near_duplicate_jaccard_max"])
    for left in range(len(query_grams)):
        for right in range(left + 1, len(query_grams)):
            score = jaccard(query_grams[left][1], query_grams[right][1])
            if score > threshold:
                problems.append(
                    f"yakın query kopyası: {query_grams[left][0]} / {query_grams[right][0]} ({score:.3f})"
                )
                if len(problems) >= 100:
                    return problems

    records = []
    for item in items:
        records.append((item["family_id"], f"query:{item['family_id']}", "query", item["query"]))
        for candidate in item["candidates"]:
            records.append((item["family_id"], candidate["id"], "candidate", candidate["text"]))
    candidate_threshold = float(cfg["quality"].get("cross_family_candidate_jaccard_max", 0.90))
    for left, right, score in _candidate_near_duplicates(records, candidate_threshold):
        if left[0] == right[0]:
            continue
        if left[2] == "query" and right[2] == "query":
            continue
        kind = "query-candidate" if left[2] != right[2] else "candidate-candidate"
        problems.append(f"cross-family yakın {kind} kopyası: {left[1]} / {right[1]} ({score:.3f})")
        if len(problems) >= 100:
            return problems

    easy_gold_threshold = float(
        cfg["quality"].get("cross_family_easy_gold_jaccard_max", 0.75)
    )
    for left, right, score in _candidate_near_duplicates(gold_easy_records, easy_gold_threshold):
        if left[0] == right[0] or {left[2], right[2]} != {"positive", "easy_negative"}:
            continue
        problems.append(
            f"başka family gold'una yakın easy: {left[1]} / {right[1]} ({score:.3f})"
        )
        if len(problems) >= 100:
            return problems

    template_members: dict[str, list[str]] = defaultdict(list)
    for item in items:
        positive = next((row for row in item["candidates"] if row.get("role") == "positive"), None)
        if positive:
            skeleton = _critical_skeleton(
                positive.get("critical_sentence", ""), positive.get("critical_word", "")
            )
            if skeleton:
                template_members[" ".join(skeleton)].append(item["family_id"])
    reuse_limit = int(cfg["quality"].get("max_abstract_template_reuse", 4))
    for signature, family_ids in template_members.items():
        if len(family_ids) > reuse_limit:
            problems.append(
                f"soyut kritik şablon {reuse_limit} kezden fazla tekrarlandı: "
                f"{family_ids[:8]} | {signature[:120]}"
            )
    return problems


def _word_shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    values = tokens(text)
    if not values:
        return set()
    if len(values) <= width:
        return {tuple(values)}
    return {tuple(values[index : index + width]) for index in range(len(values) - width + 1)}


def _candidate_near_duplicates(
    records: list[tuple[str, str, str, str]], threshold: float
) -> list[tuple[tuple, tuple, float]]:
    """Find plausible near duplicates via a word-shingle index, then verify with char Jaccard."""
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    grams = []
    for record_index, record in enumerate(records):
        grams.append(char_ngrams(record[3]))
        for shingle in _word_shingles(record[3]):
            index[shingle].append(record_index)
    candidate_pairs = set()
    for members in index.values():
        if len(members) > 80:
            continue
        for left_at in range(len(members)):
            for right_at in range(left_at + 1, len(members)):
                left, right = members[left_at], members[right_at]
                if records[left][0] != records[right][0]:
                    candidate_pairs.add((left, right))
    found = []
    for left, right in sorted(candidate_pairs):
        score = jaccard(grams[left], grams[right])
        if score > threshold:
            found.append((records[left], records[right], score))
    return found


def train_test_leakage_problems(
    test_items: list[dict[str, Any]], train_items: list[dict[str, Any]], cfg: dict[str, Any]
) -> list[str]:
    """Audit exact and fuzzy query/candidate leakage across future train and frozen test."""
    records = []
    for source, items in (("test", test_items), ("train", train_items)):
        for item in items:
            family_id = f"{source}:{item.get('family_id', item.get('id', 'unknown'))}"
            query = item.get("query")
            if isinstance(query, str):
                records.append((family_id, f"{family_id}:query", source, query))
            for candidate in item.get("candidates", []):
                text = candidate.get("text") if isinstance(candidate, dict) else None
                if isinstance(text, str):
                    records.append((family_id, f"{family_id}:{candidate.get('id', 'candidate')}", source, text))
    threshold = float(cfg["quality"].get("train_test_jaccard_max", 0.85))
    problems = []
    normalized: dict[str, set[str]] = defaultdict(set)
    for family_id, record_id, source, text in records:
        normalized[" ".join(tokens(text))].add(source)
        if len(normalized[" ".join(tokens(text))]) > 1:
            problems.append(f"train-test birebir kopya: {record_id}")
    for left, right, score in _candidate_near_duplicates(records, threshold):
        if left[2] != right[2]:
            problems.append(f"train-test yakın kopya: {left[1]} / {right[1]} ({score:.3f})")
        if len(problems) >= 200:
            break
    return sorted(set(problems))


def _bm25_scores(query: str, docs: list[str]) -> list[float]:
    doc_tokens = [tokens(doc) for doc in docs]
    query_tokens = tokens(query)
    n_docs = len(docs)
    avgdl = sum(map(len, doc_tokens)) / max(1, n_docs)
    df = Counter(token for token in set(query_tokens) for doc in doc_tokens if token in set(doc))
    scores = []
    for doc in doc_tokens:
        tf = Counter(doc)
        score = 0.0
        for term in query_tokens:
            freq = tf[term]
            if not freq:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(doc) / max(1, avgdl)))
        scores.append(score)
    return scores


def _tie_aware_top_probability(scores: list[float], gold_index: int) -> float:
    gold_score = scores[gold_index]
    if any(score > gold_score for score in scores):
        return 0.0
    return 1.0 / sum(score == gold_score for score in scores)


def artifact_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    wins = Counter()
    gold_positions = Counter()
    lexical_gaps = []
    close_hard_counts = []
    for item in items:
        candidates = item["candidates"]
        gold_index = next(index for index, candidate in enumerate(candidates) if candidate["id"] == item["gold_id"])
        gold_positions[gold_index + 1] += 1
        lengths = [len(candidate["text"]) for candidate in candidates]
        token_lengths = [len(tokens(candidate["text"])) for candidate in candidates]
        query_words = set(tokens(item["query"]))
        overlap = [len(query_words & set(tokens(candidate["text"]))) / max(1, len(query_words)) for candidate in candidates]
        query_grams = char_ngrams(item["query"])
        trigram = [jaccard(query_grams, char_ngrams(candidate["text"])) for candidate in candidates]
        bm25 = _bm25_scores(item["query"], [candidate["text"] for candidate in candidates])
        for name, scores in {
            "longest_candidate": lengths,
            "most_tokens": token_lengths,
            "word_overlap": overlap,
            "character_3gram": trigram,
            "bm25": bm25,
        }.items():
            wins[name] += _tie_aware_top_probability(scores, gold_index)
        lexical = item.get("lexical_audit") or lexical_family_audit(item)
        if lexical:
            lexical_gaps.append(float(lexical["gold_hard_median_gap"]))
            close_hard_counts.append(int(lexical["hards_at_or_above_gold"]))
    n = max(1, len(items))
    generator_prefixes: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        generator_id = str(item.get("generator_id", "unknown"))
        generator_prefixes[generator_id][" ".join(tokens(item["query"])[:3])] += 1
    return {
        "n_families": len(items),
        "chance_recall_at_1": round(1 / 11, 4),
        "recall_at_1_tie_aware": {
            name: round(value / n, 4) for name, value in sorted(wins.items())
        },
        "gold_position_counts": dict(sorted(gold_positions.items())),
        "longest_gold_rate": round(wins["longest_candidate"] / n, 4),
        "lexical_balance": {
            "mean_gold_hard_median_gap": round(sum(lexical_gaps) / max(1, len(lexical_gaps)), 4),
            "mean_hards_at_or_above_gold": round(
                sum(close_hard_counts) / max(1, len(close_hard_counts)), 4
            ),
        },
        "generator_query_prefix_top10": {
            generator: counts.most_common(10) for generator, counts in sorted(generator_prefixes.items())
        },
    }
