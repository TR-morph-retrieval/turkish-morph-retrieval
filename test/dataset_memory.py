"""Structured, aggregate-only dataset memory for coordinated generation.

The registry is deliberately external to the language models.  It stores trusted slot contracts,
compact family metadata and lifecycle events in SQLite; raw prior examples are never returned in a
generation context.  This keeps prompts small and avoids turning earlier test items into accidental
few-shot examples.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
TAG_PATTERN = re.compile(r"^[a-z0-9_]{3,64}$")
PARTICIPANT_ROLES = (
    "agent", "patient", "experiencer", "causer", "possessor", "theme", "source", "goal",
    "location", "beneficiary", "instrument", "other",
)
PARTICIPANT_BINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "role": {"type": "string", "enum": list(PARTICIPANT_ROLES)},
        "participant": {"type": "string", "minLength": 1, "maxLength": 80},
    },
    "required": ["role", "participant"],
}
POLARITIES = ("affirmative", "negative", "interrogative", "conditional", "mixed")
TEMPORAL_FRAMES = ("past", "present", "future", "habitual", "atemporal", "mixed")
SCOPE_TARGETS = (
    "predicate", "argument", "embedded_clause", "nominal_predicate", "focus", "discourse",
    "mixed",
)

SEMANTIC_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "narrative_tag": {"type": "string", "pattern": "^[a-z0-9_]{3,64}$"},
        "event_type": {"type": "string", "pattern": "^[a-z0-9_]{3,64}$"},
        "participant_roles": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {"type": "string", "enum": list(PARTICIPANT_ROLES)},
        },
        "participant_bindings": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": PARTICIPANT_BINDING_SCHEMA,
        },
        "polarity": {"type": "string", "enum": list(POLARITIES)},
        "temporal_frame": {"type": "string", "enum": list(TEMPORAL_FRAMES)},
        "scope_target": {"type": "string", "enum": list(SCOPE_TARGETS)},
    },
    "required": [
        "narrative_tag", "event_type", "participant_roles", "participant_bindings",
        "polarity", "temporal_frame", "scope_target",
    ],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tr_key(value: Any) -> str:
    return str(value or "").strip().replace("İ", "i").replace("I", "ı").casefold()


def semantic_profile_problems(profile: Any) -> list[str]:
    """Validate the small generator-declared semantic profile without a JSON-schema dependency."""
    if not isinstance(profile, dict):
        return ["semantic_profile object değil"]
    required = set(SEMANTIC_PROFILE_SCHEMA["required"])
    if set(profile) != required:
        return [
            "semantic_profile alanları eksik/fazla: "
            f"beklenen={sorted(required)} gelen={sorted(profile)}"
        ]
    problems = []
    for name in ("narrative_tag", "event_type"):
        value = profile.get(name)
        if not isinstance(value, str) or not TAG_PATTERN.fullmatch(value):
            problems.append(f"semantic_profile.{name} ASCII snake_case değil")
    roles = profile.get("participant_roles")
    if (
        not isinstance(roles, list) or not 1 <= len(roles) <= 6
        or len(roles) != len(set(roles))
        or any(role not in PARTICIPANT_ROLES for role in roles)
    ):
        problems.append("semantic_profile.participant_roles geçersiz")
    bindings = profile.get("participant_bindings")
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 6:
        problems.append("semantic_profile.participant_bindings geçersiz")
    else:
        binding_roles = []
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {"role", "participant"}:
                problems.append("semantic_profile.participant_bindings girdisi geçersiz")
                continue
            role = binding.get("role")
            participant = binding.get("participant")
            if role not in PARTICIPANT_ROLES or not isinstance(participant, str) or not participant.strip():
                problems.append("semantic_profile.participant_bindings rol/katılımcı değeri geçersiz")
                continue
            binding_roles.append(role)
        if len(binding_roles) != len(set(binding_roles)):
            problems.append("semantic_profile.participant_bindings rol tekrarı içeriyor")
        if isinstance(roles, list) and set(binding_roles) != set(roles):
            problems.append("semantic_profile participant_roles ve participant_bindings uyuşmuyor")
    if profile.get("polarity") not in POLARITIES:
        problems.append("semantic_profile.polarity geçersiz")
    if profile.get("temporal_frame") not in TEMPORAL_FRAMES:
        problems.append("semantic_profile.temporal_frame geçersiz")
    if profile.get("scope_target") not in SCOPE_TARGETS:
        problems.append("semantic_profile.scope_target geçersiz")
    return problems


def family_memory_tags(family: dict[str, Any]) -> dict[str, Any]:
    """Build the compact, model-friendly metadata attached to an accepted family."""
    profile = dict(family.get("semantic_profile") or {})
    feature = str(family.get("target_feature", ""))
    return {
        "morphology": {
            "phenomenon": feature,
            "macro": family.get("macro_phenomenon", family.get("phenomenon")),
            "objective": family.get("objective"),
            "layer": family.get("layer"),
            "feature_components": [part for part in feature.split(".") if part],
            "surface_forms": list(family.get("surface_forms") or []),
            "critical_lemma": family.get("critical_lemma"),
            "contrast_delta": family.get("feature_delta"),
        },
        "semantics": {
            "frame_id": family.get("semantic_frame_id"),
            **profile,
        },
        "generalization": {
            "split": family.get("target_split", family.get("split")),
            "bucket": family.get("generalization_bucket"),
            "tags": list(family.get("generalization_tags") or []),
            "domain": family.get("domain"),
            "register": family.get("register"),
            "template_id": family.get("template_id", family.get("template")),
            "family_mode": family.get("family_mode"),
        },
        "controls": {
            "delta": family.get("feature_delta"),
            "invariants": list((family.get("edit_script") or {}).get("invariants") or []),
        },
        "provenance": {
            "generator_id": family.get(
                "generator_id", (family.get("provenance") or {}).get("generator")
            ),
            "source_type": family.get("source_type"),
        },
    }


class DatasetMemory:
    """Transactional slot coordination and metadata/coverage registry."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialise(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS registry_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS slots (
                    slot_id TEXT PRIMARY KEY,
                    slot_index INTEGER NOT NULL,
                    target_split TEXT NOT NULL,
                    generator_id TEXT NOT NULL,
                    target_feature TEXT NOT NULL,
                    macro_phenomenon TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    register_name TEXT NOT NULL,
                    generalization_bucket TEXT NOT NULL,
                    template_id TEXT NOT NULL,
                    contract_sha256 TEXT NOT NULL,
                    contract_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'planned',
                    last_stage TEXT,
                    reservation_owner TEXT,
                    reserved_at TEXT,
                    attempt_batches INTEGER NOT NULL DEFAULT 0,
                    family_id TEXT,
                    last_error_json TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_slots_status ON slots(status);
                CREATE INDEX IF NOT EXISTS idx_slots_coverage
                    ON slots(target_feature, domain, register_name, generalization_bucket);
                CREATE TABLE IF NOT EXISTS families (
                    family_id TEXT PRIMARY KEY,
                    slot_id TEXT,
                    source_dataset TEXT NOT NULL,
                    split_name TEXT,
                    target_feature TEXT,
                    macro_phenomenon TEXT,
                    morph_chain_signature TEXT,
                    critical_lemma TEXT,
                    semantic_frame_id TEXT,
                    narrative_tag TEXT,
                    event_type TEXT,
                    participant_roles_json TEXT,
                    polarity TEXT,
                    temporal_frame TEXT,
                    scope_target TEXT,
                    domain TEXT,
                    register_name TEXT,
                    template_id TEXT,
                    generator_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_families_coverage
                    ON families(target_feature, domain, register_name, narrative_tag);
                CREATE INDEX IF NOT EXISTS idx_families_lemma ON families(critical_lemma);
                CREATE INDEX IF NOT EXISTS idx_families_frame ON families(semantic_frame_id);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slot_id TEXT,
                    event_type TEXT NOT NULL,
                    actor TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            found = connection.execute(
                "SELECT value FROM registry_meta WHERE key = 'schema_version'"
            ).fetchone()
            if found and int(found["value"]) != SCHEMA_VERSION:
                raise ValueError(
                    f"dataset memory schema {found['value']}; beklenen {SCHEMA_VERSION}"
                )
            connection.execute(
                "INSERT OR IGNORE INTO registry_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def sync_plan(self, slots: Iterable[dict[str, Any]]) -> None:
        """Insert immutable slot contracts and reject silent plan drift."""
        now = _now()
        with closing(self._connect()) as connection, connection:
            for slot in slots:
                contract = _canonical_json(slot)
                digest = _json_hash(slot)
                previous = connection.execute(
                    "SELECT contract_sha256 FROM slots WHERE slot_id = ?", (slot["slot_id"],)
                ).fetchone()
                if previous and previous["contract_sha256"] != digest:
                    raise ValueError(f"dataset memory slot contract değişti: {slot['slot_id']}")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO slots(
                        slot_id, slot_index, target_split, generator_id, target_feature,
                        macro_phenomenon, domain, register_name, generalization_bucket, template_id,
                        contract_sha256, contract_json, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        slot["slot_id"], int(slot["index"]), slot["target_split"],
                        slot["generator_id"], slot["feature"]["key"], slot["macro_phenomenon"],
                        slot["domain"], slot["register"], slot["generalization_bucket"],
                        slot["template"]["id"], digest, contract, now,
                    ),
                )

    def reserve_slot(
        self, slot_id: str, owner: str, stale_after_seconds: int = 21600
    ) -> bool:
        """Atomically reserve one slot; stale reservations may be reclaimed."""
        now = datetime.now(timezone.utc)
        stale_before = (now - timedelta(seconds=stale_after_seconds)).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, reservation_owner, reserved_at FROM slots WHERE slot_id = ?",
                (slot_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"dataset memory slot yok: {slot_id}")
            available = row["status"] in {"planned", "rejected", "failed"}
            stale = row["status"] == "reserved" and (
                not row["reserved_at"] or row["reserved_at"] < stale_before
            )
            if not (available or stale or row["reservation_owner"] == owner):
                return False
            timestamp = now.isoformat()
            connection.execute(
                """
                UPDATE slots SET status = 'reserved', reservation_owner = ?, reserved_at = ?,
                    attempt_batches = attempt_batches + 1, updated_at = ? WHERE slot_id = ?
                """,
                (owner, timestamp, timestamp, slot_id),
            )
            connection.execute(
                "INSERT INTO events(slot_id, event_type, actor, payload_json, created_at) "
                "VALUES(?, 'reserved', ?, '{}', ?)",
                (slot_id, owner, timestamp),
            )
            return True

    def record_stage(
        self, slot_id: str, stage: str, actor: str, payload: dict[str, Any] | None = None
    ) -> None:
        now = _now()
        compact_payload = payload or {}
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "UPDATE slots SET last_stage = ?, updated_at = ? WHERE slot_id = ?",
                (stage, now, slot_id),
            )
            connection.execute(
                "INSERT INTO events(slot_id, event_type, actor, payload_json, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (slot_id, stage, actor, _canonical_json(compact_payload), now),
            )

    def record_outcome(
        self, slot_id: str, status: str, record: dict[str, Any], actor: str = "pipeline"
    ) -> None:
        if status not in {"accepted", "rejected", "failed", "needs_review"}:
            raise ValueError(f"dataset memory outcome geçersiz: {status}")
        now = _now()
        family_id = record.get("family_id") if status in {"accepted", "needs_review"} else None
        error = (
            None if status == "accepted"
            else _canonical_json(record.get("problems", record.get("review_reasons", [])))
        )
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE slots SET status = ?, last_stage = ?, reservation_owner = NULL,
                    reserved_at = NULL, family_id = COALESCE(?, family_id), last_error_json = ?,
                    updated_at = ? WHERE slot_id = ?
                """,
                (status, status, family_id, error, now, slot_id),
            )
            connection.execute(
                "INSERT INTO events(slot_id, event_type, actor, payload_json, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (slot_id, status, actor, _canonical_json({"family_id": family_id}), now),
            )
            if status == "accepted":
                self._upsert_family(connection, record, "current_run", now)

    def slot_status(self, slot_id: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status FROM slots WHERE slot_id = ?", (slot_id,)
            ).fetchone()
        return str(row["status"]) if row else None

    def _upsert_family(
        self, connection: sqlite3.Connection, family: dict[str, Any], source: str, now: str
    ) -> None:
        metadata = family_memory_tags(family)
        semantics = metadata["semantics"]
        morphology = metadata["morphology"]
        generalization = metadata["generalization"]
        family_id = str(
            family.get("family_id") or family.get("query_id")
            or f"imported_{_json_hash(family)[:16]}"
        )
        connection.execute(
            """
            INSERT INTO families(
                family_id, slot_id, source_dataset, split_name, target_feature,
                macro_phenomenon, morph_chain_signature, critical_lemma, semantic_frame_id,
                narrative_tag, event_type, participant_roles_json, polarity, temporal_frame,
                scope_target, domain, register_name, template_id, generator_id, metadata_json,
                created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(family_id) DO UPDATE SET
                slot_id = excluded.slot_id,
                split_name = excluded.split_name,
                target_feature = excluded.target_feature,
                macro_phenomenon = excluded.macro_phenomenon,
                morph_chain_signature = excluded.morph_chain_signature,
                critical_lemma = excluded.critical_lemma,
                semantic_frame_id = excluded.semantic_frame_id,
                narrative_tag = excluded.narrative_tag,
                event_type = excluded.event_type,
                participant_roles_json = excluded.participant_roles_json,
                polarity = excluded.polarity,
                temporal_frame = excluded.temporal_frame,
                scope_target = excluded.scope_target,
                domain = excluded.domain,
                register_name = excluded.register_name,
                template_id = excluded.template_id,
                generator_id = excluded.generator_id,
                metadata_json = excluded.metadata_json,
                source_dataset = excluded.source_dataset,
                created_at = excluded.created_at
            """,
            (
                family_id, family.get("slot_id"), source, generalization.get("split"),
                morphology.get("phenomenon"), morphology.get("macro"),
                morphology.get("phenomenon"), morphology.get("critical_lemma"),
                semantics.get("frame_id"), semantics.get("narrative_tag"),
                semantics.get("event_type"),
                _canonical_json(semantics.get("participant_roles") or []),
                semantics.get("polarity"), semantics.get("temporal_frame"),
                semantics.get("scope_target"), generalization.get("domain"),
                generalization.get("register"), generalization.get("template_id"),
                metadata["provenance"].get("generator_id"), _canonical_json(metadata), now,
            ),
        )

    def ingest_families(self, families: Iterable[dict[str, Any]], source: str) -> int:
        """Import train/dev metadata so new generations can avoid cross-dataset reuse."""
        count = 0
        now = _now()
        with closing(self._connect()) as connection, connection:
            for family in families:
                self._upsert_family(connection, family, source, now)
                count += 1
        return count

    def generation_context(self, slot: dict[str, Any], limit: int = 24) -> dict[str, Any]:
        """Return only aggregate counts and compact tags, never previous query/candidate text."""
        feature = slot["feature"]["key"]
        domain = slot["domain"]
        register = slot["register"]
        with closing(self._connect()) as connection, connection:
            counts = {
                "accepted_total": connection.execute(
                    "SELECT COUNT(*) AS n FROM families"
                ).fetchone()["n"],
                "same_feature": connection.execute(
                    "SELECT COUNT(*) AS n FROM families WHERE target_feature = ?", (feature,)
                ).fetchone()["n"],
                "same_feature_domain": connection.execute(
                    "SELECT COUNT(*) AS n FROM families WHERE target_feature = ? AND domain = ?",
                    (feature, domain),
                ).fetchone()["n"],
                "same_feature_domain_register": connection.execute(
                    """SELECT COUNT(*) AS n FROM families
                       WHERE target_feature = ? AND domain = ? AND register_name = ?""",
                    (feature, domain, register),
                ).fetchone()["n"],
            }
            lemmas = [
                row["critical_lemma"] for row in connection.execute(
                    """SELECT DISTINCT critical_lemma FROM families
                       WHERE critical_lemma IS NOT NULL AND critical_lemma != ''
                       ORDER BY CASE WHEN target_feature = ? THEN 0 ELSE 1 END, created_at DESC
                       LIMIT ?""",
                    (feature, limit),
                )
            ]
            narratives = [
                row["narrative_tag"] for row in connection.execute(
                    """SELECT DISTINCT narrative_tag FROM families
                       WHERE narrative_tag IS NOT NULL AND narrative_tag != ''
                       ORDER BY CASE WHEN target_feature = ? THEN 0 ELSE 1 END,
                                CASE WHEN domain = ? THEN 0 ELSE 1 END, created_at DESC
                       LIMIT ?""",
                    (feature, domain, limit),
                )
            ]
        return {
            "policy": "aggregate_only_no_prior_text",
            "coverage": counts,
            "avoid_critical_lemmas": lemmas,
            "avoid_narrative_tags": narratives,
        }

    def conflicts_for(self, family: dict[str, Any]) -> list[str]:
        """Enforce the disjoint axis promised by each generalization bucket."""
        family_id = str(family.get("family_id", ""))
        frame = str(family.get("semantic_frame_id", "")).strip()
        lemma = _tr_key(family.get("critical_lemma"))
        template = str(family.get("template_id", "")).strip()
        feature = str(family.get("target_feature", "")).strip()
        bucket = family.get("generalization_bucket")
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                """SELECT family_id, source_dataset, critical_lemma, semantic_frame_id,
                          template_id, morph_chain_signature
                   FROM families WHERE family_id != ?""",
                (family_id,),
            ).fetchall()
        problems = []
        if frame and any(str(row["semantic_frame_id"] or "").strip() == frame for row in rows):
            problems.append(f"dataset memory semantic_frame tekrarı: {frame}")
        # Earlier releases of the same benchmark help lexical/narrative diversity, but are not
        # train/external data. They must not invalidate test-only holdout buckets in an extension.
        external = [
            row for row in rows
            if row["source_dataset"] not in {"current_run", "prior_test_release"}
        ]
        if bucket == "lemma_holdout" and lemma and any(
            _tr_key(row["critical_lemma"]) == lemma for row in external
        ):
            problems.append(f"lemma_holdout train/external lemma tekrarı: {family.get('critical_lemma')}")
        if bucket == "template_holdout" and template and any(
            str(row["template_id"] or "").strip() == template for row in external
        ):
            problems.append(f"template_holdout train/external template tekrarı: {template}")
        if bucket == "composition_holdout" and feature and any(
            str(row["morph_chain_signature"] or "").strip() == feature for row in external
        ):
            problems.append(f"composition_holdout train/external zincir tekrarı: {feature}")
        return problems

    def report(self) -> dict[str, Any]:
        with closing(self._connect()) as connection, connection:
            states = dict(Counter(
                row["status"] for row in connection.execute("SELECT status FROM slots")
            ))
            families = connection.execute("SELECT COUNT(*) AS n FROM families").fetchone()["n"]
            observed_cells = connection.execute(
                """SELECT COUNT(*) AS n FROM (
                       SELECT 1 FROM families
                       GROUP BY target_feature, domain, register_name, narrative_tag
                   )"""
            ).fetchone()["n"]
            coverage = {}
            for column in ("target_feature", "domain", "register_name", "narrative_tag"):
                coverage[column] = {
                    str(row["value"]): row["count"]
                    for row in connection.execute(
                        f"""SELECT {column} AS value, COUNT(*) AS count FROM families
                            WHERE {column} IS NOT NULL AND {column} != ''
                            GROUP BY {column} ORDER BY {column}"""
                    )
                }
        return {
            "schema_version": SCHEMA_VERSION,
            "database": str(self.path),
            "slot_states": states,
            "family_count": families,
            "observed_coverage_cells": observed_cells,
            "coverage": coverage,
        }
