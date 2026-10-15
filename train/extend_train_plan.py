#!/usr/bin/env python3
"""Safely extend the live 1,250-family train run to 2,250 families.

The existing 1,250 plan entries and accepted SQLite rows are preserved byte for
byte. Only new slots are appended in stable milestones. A checked-in 20-family
collaboration map is then written for the agreed producer ranges.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from production import HERE, load_config
from shards import json_sha, manifest_path, read_jsonl, sha256_bytes, write_json
from workflow import atomic_json, contract, make_plan, read_json, run_lock, verify


OWNER_RANGES = [
    ("arda", 1, 750),
    ("burak", 751, 1000),
    ("kuzey", 1001, 1250),
    ("emir", 1251, 1500),
    ("murat", 1501, 1750),
    ("burak", 1751, 2000),
    ("burak", 2001, 2250),
]

MILESTONES = (1750, 2000, 2250)


def chunked_allocations(chunk_size: int = 20):
    allocations = []
    for producer, start, end in OWNER_RANGES:
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + chunk_size - 1, end)
            allocations.append({"producer": producer, "from": cursor, "to": chunk_end})
            cursor = chunk_end + 1
    return allocations


def migrate_existing_shard_contracts(folder: Path, shard_dir: Path, manifest: dict, plan: list):
    """Retarget unchanged prefix shards after a plan-only append migration."""
    contract_value = {
        "contract_sha256": manifest["contract_sha256"],
        "config": manifest["config"].get("version"),
    }
    migrated = 0
    for shard in sorted(shard_dir.glob("*.jsonl")):
        sidecar = manifest_path(shard)
        if not sidecar.exists():
            raise ValueError(f"Shard manifesti eksik: {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("shard_sha256") != sha256_bytes(shard.read_bytes()):
            raise ValueError(f"Shard checksum uyuşmuyor: {shard}")
        start, end = metadata["from"], metadata["to"]
        expected_ids = [spec["slot_id"] for spec in plan[start - 1:end]]
        if metadata.get("slot_ids") != expected_ids or len(read_jsonl(shard)) != end - start + 1:
            raise ValueError(f"Shard genişletilen planın değişmeyen önekiyle uyuşmuyor: {shard}")
        metadata["run_contract"] = contract_value
        metadata["run_contract_sha256"] = json_sha(contract_value)
        write_json(sidecar, metadata)
        migrated += 1
    return migrated


def extend(folder: Path, shard_dir: Path, target_size: int = 2250, seed: int = 42):
    manifest, protected, plan = verify(folder)
    original_size = len(plan)
    if original_size not in {1250, *MILESTONES}:
        raise ValueError(f"Beklenen plan boyutu 1250/1750/2000/2250; bulunan {original_size}")

    for milestone in MILESTONES:
        if len(plan) >= milestone or milestone > target_size:
            continue
        before = len(plan)
        extra = make_plan(protected, milestone - before, seed + before)
        for offset, spec in enumerate(extra, start=before + 1):
            spec["slot_id"] = f"train_{offset:06d}"
        extended = [*plan, *extra]

        connection = sqlite3.connect(folder / "state.sqlite3")
        try:
            connection.execute("BEGIN IMMEDIATE")
            for spec in extra:
                connection.execute(
                    "INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                    (spec["slot_id"], json.dumps(spec, ensure_ascii=False),
                     "pending", 1, None, None, None),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        manifest["contract_sha256"] = contract(load_config(), protected, extended)
        history = manifest.setdefault("extension_history", [])
        history.append({"from": before, "to": milestone})
        manifest["extended_from"] = original_size
        manifest["extended_to"] = milestone
        atomic_json(folder / "plan.json", extended)
        atomic_json(folder / "manifest.json", manifest)
        plan = extended

    if len(plan) != target_size:
        raise ValueError(f"Plan {target_size} family'ye uzatılamadı; bulunan {len(plan)}")

    migrated_shards = migrate_existing_shard_contracts(folder, shard_dir, manifest, plan)
    allocations = chunked_allocations(20)
    expected = [i for item in allocations for i in range(item["from"], item["to"] + 1)]
    if expected != list(range(1, target_size + 1)):
        raise AssertionError("Atamalar planı tam ve sıralı kapsamıyor")
    assignment = {
        "version": 1,
        "run_id": folder.name,
        "size": target_size,
        "chunk_size": 20,
        "contract_sha256": manifest["contract_sha256"],
        "producers": list(dict.fromkeys(item[0] for item in OWNER_RANGES)),
        "owner_ranges": [
            {"producer": producer, "from": start, "to": end}
            for producer, start, end in OWNER_RANGES
        ],
        "allocations": allocations,
    }
    write_json(shard_dir / "assignments.json", assignment)
    return {"run_id": folder.name, "size": len(plan), "assignment_sha256": json_sha(assignment),
            "owner_ranges": assignment["owner_ranges"], "chunks": len(allocations),
            "migrated_shards": migrated_shards}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="train1250")
    parser.add_argument("--shard-dir", type=Path, default=HERE / "data" / "shards")
    args = parser.parse_args()
    folder = HERE / "runs" / args.run_id
    with run_lock(folder):
        result = extend(folder, args.shard_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
