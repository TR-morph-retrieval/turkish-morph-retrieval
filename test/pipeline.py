"""Generation pipeline: plan -> generate -> QC -> cascade judges -> accepted families."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .dataset_memory import DatasetMemory, family_memory_tags
from .planner import build_plan, plan_hash, plan_statistics
from .prompts import (
    ADJUDICATOR_SYSTEM,
    GENERATOR_SYSTEM,
    MORPHOLOGY_JUDGE_SYSTEM,
    PROMPT_VERSION,
    SEMANTIC_JUDGE_SYSTEM,
    build_adjudicator_prompt,
    build_generation_batch_prompt,
    build_generation_prompt,
    build_local_candidate_repair_prompt,
    build_morphology_judge_prompt,
    build_repair_prompt,
    build_semantic_judge_prompt,
)
from .providers import make_provider
from .schema import (
    ADJUDICATOR_SCHEMA,
    GENERATION_SCHEMA,
    MORPHOLOGY_JUDGE_SCHEMA,
    SEMANTIC_JUDGE_SCHEMA,
    generation_batch_schema,
    localized_repair_schema,
)
from .validators import (
    corpus_problems,
    interpret_morphology_judge,
    interpret_semantic_judges,
    normalize_family,
    quality_score,
    validate_family,
)


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunPaths:
    root: Path
    plan: Path
    manifest: Path
    accepted: Path
    rejected: Path
    failures: Path
    report: Path
    cache: Path
    memory: Path


def paths_for(run_id: str) -> RunPaths:
    root = HERE / "runs" / run_id
    return RunPaths(
        root=root,
        plan=root / "plan.json",
        manifest=root / "run_manifest.json",
        accepted=root / "accepted.jsonl",
        rejected=root / "rejected.jsonl",
        failures=root / "failures.jsonl",
        report=root / "generation_report.json",
        cache=root / "cache",
        memory=root / "dataset_memory.sqlite3",
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _pipeline_source_hashes() -> dict[str, str]:
    names = (
        "config.py", "pipeline.py", "planner.py", "prompts.py", "schema.py", "taxonomy.py",
        "validators.py", "dataset_memory.py", "providers.py", "ranges.py", "review.py",
        "judge_report.py",
    )
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in names
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        for value in values
    )
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} bozuk JSONL: {exc}") from exc
    return rows


def initialise_run(run_id: str, cfg: dict[str, Any], slots: list[dict[str, Any]]) -> RunPaths:
    paths = paths_for(run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.cache.mkdir(parents=True, exist_ok=True)
    current_hash = plan_hash(slots)
    if paths.plan.exists():
        previous = json.loads(paths.plan.read_text(encoding="utf-8"))
        if plan_hash(previous) != current_hash:
            raise ValueError(f"{run_id} mevcut planı yeni config ile uyuşmuyor; yeni run-id kullan")
    else:
        _write_json(paths.plan, slots)

    config_sha256 = hashlib.sha256(Path(cfg["_config_path"]).read_bytes()).hexdigest()
    source_hashes = _pipeline_source_hashes()
    execution_hashes = dict(source_hashes)
    # A narrowly approved transport fix retains the dataset contract. All actual
    # execution hashes are recorded separately; arbitrary source changes still fail.
    compatibility_path = HERE / "transport_compatibility.json"
    if compatibility_path.exists():
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        if source_hashes == compatibility["execution_source_sha256"]:
            source_hashes = compatibility["dataset_source_sha256"]
    manifest = {
        "run_id": run_id,
        "dataset_name": cfg["dataset_name"],
        "dataset_version": cfg["version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_start": _git_commit(),
        "prompt_version": PROMPT_VERSION,
        "pipeline_source_sha256": source_hashes,
        "execution_source_sha256": execution_hashes,
        "config_path": cfg["_config_path"],
        "config_sha256": config_sha256,
        "plan_sha256": current_hash,
        "plan_size": len(slots),
        "plan_statistics": plan_statistics(slots),
        "generation_batch_size": int(cfg["generation"]["batch_size"]),
        "generators": [
            {
                "id": spec["id"], "provider": spec["provider"], "model": spec["model"],
                "reasoning_effort": spec.get("reasoning_effort"),
                "authentication_mode": spec.get("authentication_mode"),
            }
            for spec in cfg["generation"]["generators"]
        ],
        "judges": {
            name: {
                "enabled": spec.get("enabled", True),
                "provider": spec["provider"],
                "model": spec.get("model", ""),
                "provider_preferences": spec.get("provider_preferences", {}),
            }
            for name, spec in cfg["generation"]["judges"].items()
        },
        "human_review": cfg["generation"]["human_review"],
    }
    if paths.manifest.exists():
        previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
        invariant_keys = (
            "dataset_version", "prompt_version", "pipeline_source_sha256", "config_sha256",
            "plan_sha256", "generation_batch_size", "generators", "judges", "human_review"
        )
        changed = [key for key in invariant_keys if previous.get(key) != manifest.get(key)]
        if changed:
            raise ValueError(
                f"{run_id} model/config/prompt karışımını engellemek için devam ettirilmedi; "
                f"değişen alanlar: {changed}. Yeni run-id kullan."
            )
    else:
        _write_json(paths.manifest, manifest)
    if execution_hashes != source_hashes:
        _write_json(paths.root / "transport_execution.json", {
            "policy": "length-budget-retry-v1", "git_commit": _git_commit(),
            "dataset_source_sha256": source_hashes,
            "execution_source_sha256": execution_hashes,
        })
    DatasetMemory(paths.memory).sync_plan(slots)
    return paths


def _request_provenance(response) -> dict[str, Any]:
    return {
        "execution_source_sha256": _pipeline_source_hashes(),
        "provider": response.provider,
        "model": response.model,
        "request_hash": response.request_hash,
        "cache_hit": response.cache_hit,
        "usage": response.usage,
        "actual_model": response.actual_model,
        "route_provider": response.route_provider,
    }


def _apply_local_candidate_repairs(
    previous: dict[str, Any], patch: dict[str, Any], repair_slots: list[str]
) -> dict[str, Any]:
    expected = set(repair_slots)
    rows = patch.get("candidates", [])
    received = [row.get("candidate_slot") for row in rows if isinstance(row, dict)]
    if len(received) != len(expected) or set(received) != expected:
        raise ValueError(
            f"local repair slotları uyuşmuyor; beklenen={sorted(expected)}, gelen={received}"
        )
    repaired = deepcopy(previous)
    replacements = {row["candidate_slot"]: row for row in rows}
    found = set()
    for candidate in repaired.get("candidates", []):
        slot = candidate.get("candidate_slot")
        if slot in replacements:
            candidate.update({
                "critical_sentence": replacements[slot]["critical_sentence"],
                "critical_word": replacements[slot]["critical_word"],
            })
            found.add(slot)
    if found != expected:
        raise ValueError(f"önceki family local repair slotlarını taşımıyor: {sorted(expected - found)}")
    return repaired


def _process_slot(
    slot, cfg, generators, judges, start_refill_round: int = 0, stage_callback=None,
    memory_validator=None, initial_generation=None, generator_lock=None,
) -> tuple[str, dict[str, Any]]:
    generator = generators[slot["generator_id"]]
    max_attempts = int(cfg["generation"]["max_generation_attempts"])
    refill_count = int(cfg["generation"]["refill_rounds_per_call"])
    refill_history: list[dict[str, Any]] = []
    last_raw = None
    last_family = None
    last_stage = "deterministic_validation"
    last_problems: list[str] = []

    def repair_slots_for(problems: list[str], normalized: dict[str, Any] | None) -> list[str]:
        if not normalized:
            return []
        global_markers = (
            "family naturalness", "dataset_memory", "dataset memory", "duplicate", "tekrar",
            "lexical overlap", "query ile gold", "candidate token uzunluk oranı",
            "normalizasyon hatası", "candidate kapsamı",
        )
        if any(marker in problem for problem in problems for marker in global_markers):
            return []
        slots: set[str] = set()
        for candidate in normalized.get("candidates", []):
            candidate_id = candidate.get("id", "")
            if candidate_id and any(candidate_id in problem for problem in problems):
                slots.add(candidate.get("candidate_slot", ""))
        if any("gold" in problem for problem in problems):
            slots.add("positive_01")
        return sorted(slot for slot in slots if slot)

    for refill_round in range(start_refill_round, start_refill_round + refill_count):
        # The nonce makes a replacement a fresh cached request while every balancing attribute
        # (feature, split, generator, lengths and holdout bucket) remains fixed to the slot.
        prompt_slot = {**slot, "refill_round": refill_round}
        generation_provenance = []
        previous = None
        validation_problems: list[str] = []
        family = None
        for attempt in range(max_attempts):
            repair_slots: list[str] = []
            response_schema = GENERATION_SCHEMA
            response_task = "generate_test_family"
            repair_base: dict[str, Any] = {}
            if attempt == 0 and not refill_history and initial_generation is not None:
                previous, provenance = initial_generation
                generation_provenance.append(provenance)
                initial_generation = None
                if stage_callback:
                    stage_callback(
                        "generated",
                        {"refill_round": refill_round, "attempt": attempt, "batched": True},
                    )
                try:
                    family = normalize_family(previous, slot)
                    validation_problems = validate_family(family, slot, cfg)
                except Exception as exc:
                    validation_problems = [
                        f"normalizasyon hatası: {type(exc).__name__}: {exc}"
                    ]
                    family = None
                if not validation_problems:
                    if stage_callback:
                        stage_callback(
                            "deterministic_validated", {"refill_round": refill_round}
                        )
                    break
                continue
            if attempt == 0 and not refill_history:
                prompt = build_generation_prompt(prompt_slot)
            else:
                feedback = validation_problems if attempt else last_problems
                normalized_feedback = family if attempt else last_family
                repair_slots = repair_slots_for(feedback, normalized_feedback)
                repair_base = previous or last_raw or {}
                if repair_slots:
                    prompt = build_local_candidate_repair_prompt(
                        prompt_slot, repair_base, feedback, repair_slots
                    )
                    response_schema = localized_repair_schema(repair_slots)
                    response_task = "repair_test_candidates"
                else:
                    prompt = build_repair_prompt(prompt_slot, repair_base, feedback)
                    response_schema = GENERATION_SCHEMA
                    response_task = "repair_test_family"
            if generator_lock:
                with generator_lock:
                    response = generator.call_json(
                        GENERATOR_SYSTEM, prompt, response_schema, response_task
                    )
            else:
                response = generator.call_json(
                    GENERATOR_SYSTEM, prompt, response_schema, response_task
                )
            if stage_callback:
                stage_callback("generated", {"refill_round": refill_round, "attempt": attempt})
            provenance = _request_provenance(response)
            if repair_slots:
                provenance["localized_repair_slots"] = repair_slots
                try:
                    previous = _apply_local_candidate_repairs(
                        repair_base, response.data, repair_slots
                    )
                except Exception as exc:
                    validation_problems = [
                        f"local repair birleştirme hatası: {type(exc).__name__}: {exc}"
                    ]
                    generation_provenance.append(provenance)
                    family = None
                    continue
            else:
                previous = response.data
            generation_provenance.append(provenance)
            try:
                family = normalize_family(previous, slot)
                validation_problems = validate_family(family, slot, cfg)
            except Exception as exc:
                validation_problems = [f"normalizasyon hatası: {type(exc).__name__}: {exc}"]
                family = None
            if not validation_problems:
                if stage_callback:
                    stage_callback("deterministic_validated", {"refill_round": refill_round})
                break

        last_raw = previous
        last_family = family
        if family is None or validation_problems:
            last_stage = "deterministic_validation"
            last_problems = validation_problems
            refill_history.append({
                "refill_round": refill_round,
                "stage": last_stage,
                "problems": validation_problems,
                "generator_attempts": generation_provenance,
            })
            continue

        memory_problems = memory_validator(family) if memory_validator else []
        if memory_problems:
            last_stage = "dataset_memory"
            last_problems = memory_problems
            refill_history.append({
                "refill_round": refill_round,
                "stage": last_stage,
                "problems": memory_problems,
                "generator_attempts": generation_provenance,
            })
            if stage_callback:
                stage_callback("dataset_memory_rejected", {"refill_round": refill_round})
            continue

        semantic_responses = []
        semantic_verdicts = []
        for permutation in cfg["generation"]["judges"]["semantic"]["permutations"]:
            response = judges["semantic"].call_json(
                SEMANTIC_JUDGE_SYSTEM,
                build_semantic_judge_prompt(family, permutation),
                SEMANTIC_JUDGE_SCHEMA,
                f"semantic_judge_{permutation}",
            )
            semantic_responses.append(_request_provenance(response))
            semantic_verdicts.append(response.data)
        semantic_problems, semantic_metadata = interpret_semantic_judges(
            family, semantic_verdicts, cfg
        )

        morphology_response = judges["morphology"].call_json(
            MORPHOLOGY_JUDGE_SYSTEM,
            build_morphology_judge_prompt(family),
            MORPHOLOGY_JUDGE_SCHEMA,
            "morphology_judge",
        )
        morphology_problems, morphology_metadata = interpret_morphology_judge(
            family, morphology_response.data, cfg
        )
        judge_problems = sorted(set(semantic_problems + morphology_problems))
        judging_provenance = {
            "semantic": semantic_responses,
            "morphology": _request_provenance(morphology_response),
        }
        judging_metadata = {
            "semantic": semantic_metadata,
            "morphology": morphology_metadata,
        }
        if stage_callback:
            stage_callback(
                "cascade_judges_completed",
                {"refill_round": refill_round, "accepted": not bool(judge_problems)},
            )

        adjudicator_record = None
        if judge_problems and judges.get("adjudicator") is not None:
            adjudicator_response = judges["adjudicator"].call_json(
                ADJUDICATOR_SYSTEM,
                build_adjudicator_prompt(
                    family, semantic_verdicts, morphology_response.data, judge_problems
                ),
                ADJUDICATOR_SCHEMA,
                "judge_disagreement_advisory",
            )
            adjudicator_record = {
                "verdict": adjudicator_response.data,
                "provenance": _request_provenance(adjudicator_response),
            }
            judging_metadata["adjudicator"] = adjudicator_response.data
            judging_provenance["adjudicator"] = adjudicator_record["provenance"]

        family["provenance"] = {
            "generator_id": slot["generator_id"],
            "refill_round": refill_round,
            "rejected_replacements": refill_history,
            "generator_attempts": generation_provenance,
            "judges": judging_provenance,
            "prompt_version": PROMPT_VERSION,
        }
        family["qc"] = {
            "deterministic": "pass",
            "judging": judging_metadata,
        }
        priority_reasons: list[str] = []
        semantic_quality_floor = int(cfg["quality"]["semantic_quality_confidence_min"])
        review_confidence_floor = int(cfg["quality"]["human_review_confidence_floor"])
        for index, semantic_pass in enumerate(semantic_metadata.get("passes", []), start=1):
            confidence = semantic_pass.get("confidence", 0)
            semantic_finding = bool(
                semantic_pass.get("support_mismatches")
                or semantic_pass.get("unnatural_candidate_ids")
                or semantic_pass.get("internally_inconsistent_candidates")
                or semantic_pass.get("length_or_style_artifact")
                or int(semantic_pass.get("family_naturalness", 0))
                < int(cfg["quality"]["judge_naturalness_min"])
            )
            if not isinstance(confidence, int) or confidence < review_confidence_floor:
                priority_reasons.append(f"semantic_{index}:very_low_confidence:{confidence}")
            elif confidence < semantic_quality_floor and semantic_finding:
                priority_reasons.append(f"semantic_{index}:low_confidence_with_finding:{confidence}")
            if semantic_pass.get("abstain"):
                priority_reasons.append(f"semantic_{index}:abstain")
            if semantic_pass.get("support_mismatches") and not semantic_pass.get(
                "relevance_authoritative", False
            ):
                priority_reasons.append(f"semantic_{index}:advisory_relevance_mismatch")
            if (
                semantic_pass.get("unnatural_candidate_ids")
                or semantic_pass.get("internally_inconsistent_candidates")
            ) and not semantic_pass.get("quality_authoritative", False):
                priority_reasons.append(f"semantic_{index}:advisory_quality_finding")

        morphology_floor = int(cfg["quality"]["morphology_judge_confidence_min"])
        morphology_confidence = morphology_metadata.get("confidence", 0)
        morphology_finding = bool(
            morphology_metadata.get("morphology_failures")
            or morphology_metadata.get("unclear_candidate_ids")
            or morphology_metadata.get("gold_target_status") != "matches_target"
            or morphology_metadata.get("allomorph_treated_as_wrong")
        )
        if (
            not isinstance(morphology_confidence, int)
            or morphology_confidence < review_confidence_floor
        ):
            priority_reasons.append(f"morphology:very_low_confidence:{morphology_confidence}")
        elif morphology_confidence < morphology_floor and morphology_finding:
            priority_reasons.append(
                f"morphology:low_confidence_with_finding:{morphology_confidence}"
            )
        if morphology_metadata.get("abstain"):
            priority_reasons.append("morphology:abstain")
        if morphology_metadata.get("unclear_candidate_ids"):
            priority_reasons.append("morphology:unclear_candidates")
        if (
            morphology_metadata.get("gold_target_status") != "matches_target"
            and not morphology_metadata.get("authoritative", False)
        ):
            priority_reasons.append("morphology:gold_not_confirmed_advisory")
        family["qc"]["human_review_priority"] = bool(priority_reasons)
        family["qc"]["human_review_priority_reasons"] = sorted(set(priority_reasons))
        family["qc"]["quality_score"] = round(quality_score(family), 5)
        family["memory_tags"] = family_memory_tags(family)

        if judge_problems:
            # A judge failure is a generation failure, not a queue for people during production.
            # The fixed slot is refilled until it passes the automatic cascade.
            refill_history.append({
                "refill_round": refill_round,
                "stage": "cascade_judges",
                "problems": judge_problems,
                "adjudicator": adjudicator_record,
            })
            last_stage, last_problems, last_family = "cascade_judges", judge_problems, family
            continue

        family["source_type"] = "llm_generated_cascade_judged"
        return "accepted", family

    return "rejected", {
        "slot_id": slot["slot_id"],
        "stage": last_stage,
        "problems": last_problems,
        "last_raw": last_raw,
        "last_family": last_family,
        "next_refill_round": start_refill_round + refill_count,
        "refill_history": refill_history,
    }


def _group_slots_by_generator(slots: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Create homogeneous batches while alternating the two generators."""
    queues: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        queues.setdefault(slot["generator_id"], []).append(slot)
    groups: list[list[dict[str, Any]]] = []
    while any(queues.values()):
        for generator_id in queues:
            if queues[generator_id]:
                groups.append(queues[generator_id][:batch_size])
                del queues[generator_id][:batch_size]
    return groups


def _process_generation_batch(entries, cfg, generators, judges, memory_validator, generator_lock):
    """Generate three families together, then validate/judge/refill each independently."""
    prompt_slots = [entry["prompt_slot"] for entry in entries]
    generator_id = prompt_slots[0]["generator_id"]
    generator = generators[generator_id]
    with generator_lock:
        response = generator.call_json(
            GENERATOR_SYSTEM,
            build_generation_batch_prompt(prompt_slots),
            generation_batch_schema(len(prompt_slots)),
            f"generate_test_batch_{prompt_slots[0]['slot_id']}",
        )
    raw_families = response.data.get("families", [])
    by_frame = {
        row.get("semantic_frame_id"): row
        for row in raw_families
        if isinstance(row, dict) and row.get("semantic_frame_id")
    }
    shared_provenance = _request_provenance(response)
    results = []
    for index, entry in enumerate(entries):
        slot = entry["prompt_slot"]
        raw = by_frame.get(slot["semantic_frame_id"])
        initial = None
        if raw is not None:
            initial = (
                raw,
                {
                    **shared_provenance,
                    "shared_batch_request": True,
                    "batch_size": len(entries),
                    "batch_member_index": index,
                },
            )
        try:
            status, record = _process_slot(
                slot,
                cfg,
                generators,
                judges,
                entry["start_refill_round"],
                entry["stage_callback"],
                memory_validator,
                initial,
                generator_lock,
            )
        except Exception as exc:
            status = "failed"
            record = {
                "slot_id": entry["slot"]["slot_id"],
                "stage": "exception",
                "problems": [f"{type(exc).__name__}: {exc}"],
            }
        results.append((entry["slot"], entry["owner"], status, record))
    return results


def _resume_refill_rounds(rejected: list[dict[str, Any]]) -> dict[str, int]:
    rounds: dict[str, int] = {}
    for row in rejected:
        slot_id = row.get("slot_id")
        if slot_id:
            rounds[slot_id] = max(rounds.get(slot_id, 0), int(row.get("next_refill_round", 0)))
    return rounds


def current_accepted(run_id: str) -> list[dict[str, Any]]:
    """Return the latest accepted record for slots whose current memory state is accepted."""
    paths = paths_for(run_id)
    memory = DatasetMemory(paths.memory)
    latest = {}
    for row in read_jsonl(paths.accepted):
        if row.get("slot_id"):
            latest[row["slot_id"]] = row
    return [
        latest[slot_id] for slot_id in sorted(latest)
        if memory.slot_status(slot_id) == "accepted"
    ]


def generate(
    run_id: str,
    config_path: str | None = None,
    limit: int | None = None,
    workers: int | None = None,
    generator_id: str | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=True)
    all_slots = build_plan(cfg)
    if offset < 0:
        raise ValueError("--offset negatif olamaz")
    if offset and limit is None:
        raise ValueError("--offset kullanırken --limit de verilmelidir")
    known_generator_ids = {spec["id"] for spec in cfg["generation"]["generators"]}
    if generator_id:
        if generator_id not in known_generator_ids:
            raise ValueError(f"bilinmeyen generator-id: {generator_id}")
    eligible_slots = [
        slot for slot in all_slots
        if generator_id is None or slot["generator_id"] == generator_id
    ]
    end = offset + limit if limit is not None else len(eligible_slots)
    if end > len(eligible_slots):
        raise ValueError("seçilen generator/slot aralığı planı aşıyor")
    slots = eligible_slots[offset:end]
    paths = initialise_run(run_id, cfg, all_slots)
    memory = DatasetMemory(paths.memory)
    accepted_before = current_accepted(run_id)
    rejected_before = read_jsonl(paths.rejected)
    completed = {row.get("slot_id") for row in accepted_before if row.get("slot_id")}
    refill_rounds = _resume_refill_rounds(rejected_before)
    pending = [
        slot
        for slot in slots
        if slot["slot_id"] not in completed
    ]
    metadata = {"dataset_version": cfg["version"], "prompt_version": PROMPT_VERSION}
    cli_workdir = paths.root / "cli_workdir"
    cli_workdir.mkdir(parents=True, exist_ok=True)
    required_generator_ids = {slot["generator_id"] for slot in pending}
    generator_specs = [
        spec for spec in cfg["generation"]["generators"]
        if spec["id"] in required_generator_ids
    ]
    generators = {
        spec["id"]: make_provider(
            {**spec, "workdir": str(cli_workdir / spec["id"])},
            paths.cache / spec["id"], metadata,
        )
        for spec in generator_specs
    }
    for spec in generator_specs:
        (cli_workdir / spec["id"]).mkdir(parents=True, exist_ok=True)
    judge_specs = cfg["generation"]["judges"]
    judges = {
        "semantic": make_provider(judge_specs["semantic"], paths.cache / "semantic_judge", metadata),
        "morphology": make_provider(
            judge_specs["morphology"], paths.cache / "morphology_judge", metadata
        ),
    }
    if judge_specs["adjudicator"].get("enabled", False):
        judges["adjudicator"] = make_provider(
            judge_specs["adjudicator"], paths.cache / "adjudicator", metadata
        )
    write_lock = threading.Lock()
    counts = Counter()
    errors = []
    started = time.time()

    worker_count = int(workers or cfg["generation"].get("workers", 4))
    batch_size = int(cfg["generation"]["batch_size"])
    submitted = 0
    generation_batches = 0
    reservation_skips = 0
    pending_batches = iter(_group_slots_by_generator(pending, batch_size))
    generator_locks = {generator_id: threading.Lock() for generator_id in generators}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        future_to_entries: dict[Any, list[dict[str, Any]]] = {}

        def fill_workers() -> None:
            nonlocal submitted, generation_batches, reservation_skips
            while len(future_to_entries) < max(1, worker_count):
                try:
                    slot_batch = next(pending_batches)
                except StopIteration:
                    return
                entries = []
                for slot in slot_batch:
                    owner = f"{run_id}:{os.getpid()}:{slot['slot_id']}"
                    if not memory.reserve_slot(slot["slot_id"], owner):
                        reservation_skips += 1
                        continue
                    prompt_slot = {
                        **slot,
                        "dataset_memory": memory.generation_context(slot),
                    }

                    def stage_callback(stage, payload, *, _slot=slot, _owner=owner):
                        memory.record_stage(_slot["slot_id"], stage, _owner, payload)

                    entries.append({
                        "slot": slot,
                        "prompt_slot": prompt_slot,
                        "owner": owner,
                        "stage_callback": stage_callback,
                        "start_refill_round": refill_rounds.get(slot["slot_id"], 0),
                    })
                if not entries:
                    continue
                generator_id = entries[0]["slot"]["generator_id"]
                future = executor.submit(
                    _process_generation_batch,
                    entries,
                    cfg,
                    generators,
                    judges,
                    memory.conflicts_for,
                    generator_locks[generator_id],
                )
                future_to_entries[future] = entries
                submitted += len(entries)
                generation_batches += 1

        fill_workers()
        while future_to_entries:
            done, _ = wait(set(future_to_entries), return_when=FIRST_COMPLETED)
            for future in done:
                entries = future_to_entries.pop(future)
                try:
                    outcomes = future.result()
                except Exception as exc:  # transient/provider failures are retryable on the next resume
                    outcomes = []
                    for entry in entries:
                        record = {
                            "slot_id": entry["slot"]["slot_id"],
                            "stage": "exception",
                            "problems": [f"{type(exc).__name__}: {exc}"],
                        }
                        errors.append(record)
                        outcomes.append((entry["slot"], entry["owner"], "failed", record))
                for slot, owner, status, record in outcomes:
                    target_path = {
                        "accepted": paths.accepted,
                        "rejected": paths.rejected,
                        "failed": paths.failures,
                    }[status]
                    _append_jsonl(target_path, record, write_lock)
                    memory.record_outcome(slot["slot_id"], status, record, actor=owner)
                    counts[status] += 1
            fill_workers()

    accepted = read_jsonl(paths.accepted)
    accepted = current_accepted(run_id)
    rejected = read_jsonl(paths.rejected)
    failures = read_jsonl(paths.failures)
    accepted_slot_ids = {item.get("slot_id") for item in accepted}
    target_count = len(slots)
    report = {
        "run_id": run_id,
        "elapsed_seconds_this_call": round(time.time() - started, 2),
        "pending_processed_this_call": submitted,
        "generation_batch_size": batch_size,
        "generation_batches_submitted_this_call": generation_batches,
        "reservation_skips_this_call": reservation_skips,
        "outcomes_this_call": dict(counts),
        "accepted_total": len(accepted),
        "target_total": target_count,
        "unfilled_slots": sum(slot["slot_id"] not in accepted_slot_ids for slot in slots),
        "complete": all(slot["slot_id"] in accepted_slot_ids for slot in slots),
        "rejected_attempt_batches_total": len(rejected),
        "rejected_slots_ever": len({item.get("slot_id") for item in rejected}),
        "retryable_failures_total": len(failures),
        "accepted_by_bucket": dict(Counter(item["generalization_bucket"] for item in accepted)),
        "accepted_by_query_sentence_count": dict(
            Counter(str(item["query_sentence_count"]) for item in accepted)
        ),
        "accepted_by_passage_sentence_count": dict(
            Counter(str(item["passage_sentence_count"]) for item in accepted)
        ),
        "accepted_by_generator": dict(Counter(item["generator_id"] for item in accepted)),
        "accepted_strict_minimal_pairs": sum(bool(item["strict_minimal_pair"]) for item in accepted),
        "rejection_stages": dict(Counter(item.get("stage", "unknown") for item in rejected)),
        "uncaught_errors_this_call": errors,
        "dataset_memory": memory.report(),
    }
    _write_json(paths.report, report)
    return report


def write_plan(run_id: str, config_path: str | None = None, size: int | None = None) -> dict[str, Any]:
    cfg = load_config(config_path, runtime=False)
    slots = build_plan(cfg, size=size)
    paths = paths_for(run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    _write_json(paths.plan, slots)
    DatasetMemory(paths.memory).sync_plan(slots)
    report = {"run_id": run_id, "size": len(slots), "sha256": plan_hash(slots), "statistics": plan_statistics(slots)}
    _write_json(paths.root / "plan_report.json", report)
    return report


def sync_shared_shards(
    run_id: str, input_dir: str | Path, config_path: str | None = None
) -> dict[str, Any]:
    """Rebuild local accepted state and SQLite memory from Git-tracked contributor shards."""
    cfg = load_config(config_path, runtime=False)
    slots = build_plan(cfg)
    paths = initialise_run(run_id, cfg, slots)
    slot_by_id = {slot["slot_id"]: slot for slot in slots}
    sources = sorted(Path(input_dir).glob("*.jsonl"))
    combined: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}

    # Keep locally generated, not-yet-exported records while bringing in teammates' shards.
    inputs = [(str(paths.accepted), row) for row in read_jsonl(paths.accepted)]
    for source in sources:
        inputs.extend((str(source), row) for row in read_jsonl(source))

    for source, family in inputs:
        slot_id = family.get("slot_id")
        if slot_id not in slot_by_id:
            raise ValueError(f"{source}: planda bulunmayan slot_id: {slot_id}")
        slot = slot_by_id[slot_id]
        if family.get("generator_id") != slot["generator_id"]:
            raise ValueError(
                f"{source}: {slot_id} generator uyuşmazlığı; "
                f"plan={slot['generator_id']} veri={family.get('generator_id')}"
            )
        problems = validate_family(family, slot, cfg)
        if problems:
            raise ValueError(f"{source}: {slot_id} deterministic QC geçmedi: {problems[:5]}")
        previous = combined.get(slot_id)
        if previous is not None and json.dumps(previous, sort_keys=True) != json.dumps(
            family, sort_keys=True
        ):
            raise ValueError(
                f"{slot_id} iki farklı içerikle paylaşılmış: {origins[slot_id]} ve {source}"
            )
        combined[slot_id] = family
        origins[slot_id] = source

    ordered = [combined[slot["slot_id"]] for slot in slots if slot["slot_id"] in combined]
    cross_family_problems = corpus_problems(ordered, cfg)
    if cross_family_problems:
        raise ValueError(
            "shared shard birleşimi cross-family QC geçmedi: "
            f"{cross_family_problems[:10]}"
        )
    _write_jsonl(paths.accepted, ordered)
    memory = DatasetMemory(paths.memory)
    for family in ordered:
        memory.record_outcome(
            family["slot_id"], "accepted", family, actor="shared_shard_sync"
        )
    return {
        "run_id": run_id,
        "shard_directory": str(Path(input_dir)),
        "shard_files": len(sources),
        "accepted_synced": len(ordered),
        "remaining": len(slots) - len(ordered),
        "memory": memory.report(),
    }


def export_shared_shard(
    run_id: str,
    output: str | Path,
    generator_id: str,
    offset: int,
    limit: int,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Export exactly one contributor's assigned generator slice as a merge-friendly shard."""
    if offset < 0 or limit < 1:
        raise ValueError("offset >= 0 ve limit >= 1 olmalıdır")
    cfg = load_config(config_path, runtime=False)
    slots = [
        slot for slot in build_plan(cfg) if slot["generator_id"] == generator_id
    ]
    selected = slots[offset:offset + limit]
    if len(selected) != limit:
        raise ValueError("shard aralığı generator kotasını aşıyor")
    accepted = {row["slot_id"]: row for row in current_accepted(run_id)}
    missing = [slot["slot_id"] for slot in selected if slot["slot_id"] not in accepted]
    if missing:
        raise ValueError(f"shard henüz tamamlanmadı; eksik slot: {missing[:5]}")
    rows = [accepted[slot["slot_id"]] for slot in selected]
    destination = Path(output)
    _write_jsonl(destination, rows)
    return {
        "run_id": run_id,
        "output": str(destination),
        "generator_id": generator_id,
        "offset": offset,
        "count": len(rows),
    }


def default_run_id() -> str:
    return datetime.now().strftime("test_%Y%m%d_%H%M%S")
