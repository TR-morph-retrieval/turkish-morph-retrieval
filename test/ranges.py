"""Sequential, Git-sharded generation ranges for multiple machines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import current_accepted, export_shared_shard, generate, paths_for, sync_shared_shards
from .planner import build_plan


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_RUN_ID = "final_v39"
DEFAULT_SHARD_DIR = REPO_ROOT / "test" / "data" / "final_shards"
PRODUCERS = {"codex": "generator_a", "claude": "generator_b"}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_path(shard: Path) -> Path:
    return shard.with_suffix(shard.suffix + ".manifest.json")


def _run_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset_version", "prompt_version", "pipeline_source_sha256", "config_sha256",
        "plan_sha256", "generation_batch_size", "generators", "judges", "human_review",
    )
    return {key: manifest.get(key) for key in keys}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _queue(generator_id: str, config_path: str | None = None) -> list[dict[str, Any]]:
    cfg = load_config(config_path, runtime=False)
    return [slot for slot in build_plan(cfg) if slot["generator_id"] == generator_id]


def validate_shared_ranges(
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
    run_id: str = DEFAULT_RUN_ID,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Verify every shard+manifest pair, exact slots, continuity and common run contract."""
    root = Path(shard_dir)
    jsonl_files = sorted(root.glob("*.jsonl"))
    manifest_files = sorted(root.glob("*.jsonl.manifest.json"))
    expected_manifests = {_manifest_path(path) for path in jsonl_files}
    orphan_shards = [str(path) for path in jsonl_files if _manifest_path(path) not in manifest_files]
    orphan_manifests = [str(path) for path in manifest_files if path not in expected_manifests]
    if orphan_shards or orphan_manifests:
        raise ValueError(
            f"shard/manifest çifti eksik; shards={orphan_shards}, manifests={orphan_manifests}"
        )

    queues = {generator_id: _queue(generator_id, config_path) for generator_id in PRODUCERS.values()}
    records: list[dict[str, Any]] = []
    contract_hashes: set[str] = set()
    for shard in jsonl_files:
        manifest = json.loads(_manifest_path(shard).read_text(encoding="utf-8"))
        rows = _read_jsonl(shard)
        producer = manifest.get("producer")
        generator_id = PRODUCERS.get(producer)
        start, end = manifest.get("from"), manifest.get("to")
        if manifest.get("run_id") != run_id:
            raise ValueError(f"{shard}: run_id uyuşmuyor")
        if generator_id is None or manifest.get("generator_id") != generator_id:
            raise ValueError(f"{shard}: producer/generator uyuşmuyor")
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start or end > 300:
            raise ValueError(f"{shard}: geçersiz 1-based inclusive aralık {start}-{end}")
        expected_slots = queues[generator_id][start - 1:end]
        expected_ids = [slot["slot_id"] for slot in expected_slots]
        actual_ids = [row.get("slot_id") for row in rows]
        checks = {
            "filename": manifest.get("shard_file") == shard.name,
            "shard_sha256": manifest.get("shard_sha256") == _sha256(shard.read_bytes()),
            "family_count": manifest.get("family_count") == end - start + 1 == len(rows),
            "slot_ids": manifest.get("slot_ids") == actual_ids == expected_ids,
            "unique_slots": len(actual_ids) == len(set(actual_ids)),
            "run_contract": manifest.get("run_contract_sha256") == _json_sha256(
                manifest.get("run_contract")
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{shard}: manifest doğrulaması başarısız: {failed}")
        contract_hashes.add(manifest["run_contract_sha256"])
        records.append({**manifest, "path": str(shard)})

    if len(contract_hashes) > 1:
        raise ValueError("shard'lar farklı config/plan/prompt/kod sözleşmeleriyle üretilmiş")
    coverage = {}
    for producer, generator_id in PRODUCERS.items():
        rows = sorted(
            (row for row in records if row["generator_id"] == generator_id),
            key=lambda row: row["from"],
        )
        expected_start = 1
        for row in rows:
            if row["from"] != expected_start:
                raise ValueError(
                    f"{producer} shard sırası boşluklu/çakışık: "
                    f"beklenen başlangıç={expected_start}, bulunan={row['from']}"
                )
            expected_start = row["to"] + 1
        coverage[producer] = expected_start - 1
    return {
        "run_id": run_id,
        "shard_dir": str(root),
        "shards": len(records),
        "coverage": coverage,
        "run_contract_sha256": next(iter(contract_hashes), None),
    }


def range_status(
    producer: str,
    start: int,
    end: int,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
    run_id: str = DEFAULT_RUN_ID,
    config_path: str | None = None,
) -> dict[str, Any]:
    if producer not in PRODUCERS:
        raise ValueError(f"producer codex veya claude olmalıdır: {producer}")
    if start < 1 or end < start or end > 300:
        raise ValueError("aralık 1-based inclusive ve 1 <= from <= to <= 300 olmalıdır")
    shared = validate_shared_ranges(shard_dir, run_id, config_path)
    expected_start = int(shared["coverage"][producer]) + 1
    if start != expected_start:
        raise ValueError(
            f"{producer} için sıradaki aralık {expected_start}'den başlamalı; "
            f"{start} çakışma veya boşluk oluşturur"
        )
    generator_id = PRODUCERS[producer]
    width = end - start + 1
    filename = f"{producer}_{start:03d}_{end:03d}.jsonl"
    return {
        "producer": producer,
        "generator_id": generator_id,
        "from": start,
        "to": end,
        "count": width,
        "offset_internal": start - 1,
        "run_id": run_id,
        "output": str(Path(shard_dir) / filename),
        "shared_before": shared,
    }


def _write_range_manifest(status: dict[str, Any]) -> dict[str, Any]:
    shard = Path(status["output"])
    rows = _read_jsonl(shard)
    run_manifest = json.loads(paths_for(status["run_id"]).manifest.read_text(encoding="utf-8"))
    contract = _run_contract(run_manifest)
    manifest = {
        "manifest_version": 1,
        "run_id": status["run_id"],
        "producer": status["producer"],
        "generator_id": status["generator_id"],
        "from": status["from"],
        "to": status["to"],
        "family_count": len(rows),
        "shard_file": shard.name,
        "slot_ids": [row.get("slot_id") for row in rows],
        "shard_sha256": _sha256(shard.read_bytes()),
        "run_contract": contract,
        "run_contract_sha256": _json_sha256(contract),
    }
    _write_json(_manifest_path(shard), manifest)
    return manifest


def _recover_local_orphan(
    producer: str,
    start: int,
    end: int,
    shard_dir: str | Path,
    run_id: str,
    config_path: str | None,
) -> bool:
    """Recover a crash between atomic shard export and its manifest write on the same machine."""
    shard = Path(shard_dir) / f"{producer}_{start:03d}_{end:03d}.jsonl"
    manifest = _manifest_path(shard)
    if not shard.exists() or manifest.exists():
        return False
    run_paths = paths_for(run_id)
    if not run_paths.manifest.exists() or not run_paths.accepted.exists():
        return False
    generator_id = PRODUCERS[producer]
    queue = _queue(generator_id, config_path)
    expected_ids = [slot["slot_id"] for slot in queue[start - 1:end]]
    rows = _read_jsonl(shard)
    if [row.get("slot_id") for row in rows] != expected_ids:
        return False
    accepted = {row.get("slot_id"): row for row in current_accepted(run_id)}
    if any(accepted.get(row["slot_id"]) != row for row in rows):
        return False
    _write_range_manifest({
        "run_id": run_id,
        "producer": producer,
        "generator_id": generator_id,
        "from": start,
        "to": end,
        "output": str(shard),
    })
    return True


def run_range(
    producer: str,
    start: int,
    end: int,
    shard_dir: str | Path = DEFAULT_SHARD_DIR,
    run_id: str = DEFAULT_RUN_ID,
    config_path: str | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Validate Git shards, sync SQLite, generate the next contiguous range, and export it."""
    _recover_local_orphan(producer, start, end, shard_dir, run_id, config_path)
    status = range_status(producer, start, end, shard_dir, run_id, config_path)
    sync_report = sync_shared_shards(run_id, shard_dir, config_path)
    local_manifest = json.loads(paths_for(run_id).manifest.read_text(encoding="utf-8"))
    local_contract_hash = _json_sha256(_run_contract(local_manifest))
    previous_contract_hash = status["shared_before"].get("run_contract_sha256")
    if previous_contract_hash and previous_contract_hash != local_contract_hash:
        raise ValueError(
            "yerel config/plan/prompt/kod sözleşmesi önceki shard'lardan farklı; "
            "aynı frozen üretim koduyla devam edin"
        )
    generation_report = generate(
        run_id,
        config_path,
        status["count"],
        workers,
        status["generator_id"],
        status["offset_internal"],
    )
    export_report = None
    manifest = None
    if generation_report["complete"]:
        export_report = export_shared_shard(
            run_id,
            status["output"],
            status["generator_id"],
            status["offset_internal"],
            status["count"],
            config_path,
        )
        manifest = _write_range_manifest(status)
        validate_shared_ranges(shard_dir, run_id, config_path)
    return {
        "range": {key: status[key] for key in ("producer", "from", "to", "count", "output")},
        "sync": sync_report,
        "generation": generation_report,
        "export": export_report,
        "manifest": manifest,
        "next_action": (
            f"{status['output']} ve {_manifest_path(Path(status['output']))} dosyalarını commit/push et"
            if manifest else "aynı range-run komutunu yeniden çalıştır; yalnız eksik slotlar refill olur"
        ),
    }
