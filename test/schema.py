"""Small provider-facing schemas; trusted metadata is added by Python."""

from __future__ import annotations

from .dataset_memory import SEMANTIC_PROFILE_SCHEMA


CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_slot": {"type": "string", "pattern": "^(positive_01|hard_0[1-8]|easy_0[1-2])$"},
        "critical_sentence": {"type": "string", "minLength": 8},
        "critical_word": {"type": "string", "minLength": 1},
    },
    "required": ["candidate_slot", "critical_sentence", "critical_word"],
}


GENERATION_SCHEMA = {
    "name": "turkish_morph_contrast_family",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_frame_id": {"type": "string"},
            "semantic_profile": SEMANTIC_PROFILE_SCHEMA,
            "critical_lemma": {"type": "string", "minLength": 2},
            "critical_word_query": {"type": "string", "minLength": 2},
            "query": {"type": "string", "minLength": 8},
            "context_sentences": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "minLength": 8},
            },
            "candidates": {
                "type": "array",
                "minItems": 11,
                "maxItems": 11,
                "items": CANDIDATE_SCHEMA,
            },
        },
        "required": [
            "semantic_frame_id", "semantic_profile", "critical_lemma", "critical_word_query", "query",
            "context_sentences", "candidates",
        ],
    },
}


def generation_batch_schema(size: int) -> dict:
    """Structured-output contract for one provider call containing independent families."""
    if size < 1:
        raise ValueError("generation batch size en az 1 olmalı")
    return {
        "name": "turkish_morph_contrast_family_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "families": {
                    "type": "array",
                    "minItems": size,
                    "maxItems": size,
                    "items": GENERATION_SCHEMA["schema"],
                }
            },
            "required": ["families"],
        },
    }


SEMANTIC_JUDGE_SCHEMA = {
    "name": "blind_semantic_retrieval_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "fully_relevant_candidate_ids": {
                "type": "array", "maxItems": 11, "items": {"type": "string"},
            },
            "unnatural_candidate_ids": {
                "type": "array", "maxItems": 11, "items": {"type": "string"},
            },
            "internally_inconsistent_candidate_ids": {
                "type": "array", "maxItems": 11, "items": {"type": "string"},
            },
            "length_or_style_artifact": {"type": "boolean"},
            "family_naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "abstain": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "fully_relevant_candidate_ids", "unnatural_candidate_ids",
            "internally_inconsistent_candidate_ids", "length_or_style_artifact",
            "family_naturalness", "confidence", "abstain", "notes",
        ],
    },
}


MORPHOLOGY_JUDGE_SCHEMA = {
    "name": "feature_aware_morphology_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_matching_candidate_ids": {
                "type": "array", "maxItems": 11, "items": {"type": "string"},
            },
            "morphologically_invalid_candidate_ids": {
                "type": "array", "maxItems": 11, "items": {"type": "string"},
            },
            "unclear_candidate_ids": {
                "type": "array", "maxItems": 11, "items": {"type": "string"},
            },
            "allomorph_treated_as_wrong": {"type": "boolean"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "abstain": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "target_matching_candidate_ids", "morphologically_invalid_candidate_ids",
            "unclear_candidate_ids", "allomorph_treated_as_wrong",
            "confidence", "abstain", "notes",
        ],
    },
}


ADJUDICATOR_SCHEMA = {
    "name": "judge_disagreement_advisory",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recommendation": {
                "type": "string", "enum": ["accept", "reject", "human_review"],
            },
            "answers_query": {"type": "array", "items": {"type": "string"}},
            "morphology_valid": {"type": "boolean"},
            "naturalness_valid": {"type": "boolean"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "reason_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "semantic_disagreement", "order_instability", "morphology_disagreement",
                        "naturalness_disagreement", "low_confidence", "other",
                    ],
                },
            },
            "notes": {"type": "string"},
        },
        "required": [
            "recommendation", "answers_query", "morphology_valid", "naturalness_valid",
            "confidence", "reason_codes", "notes",
        ],
    },
}
