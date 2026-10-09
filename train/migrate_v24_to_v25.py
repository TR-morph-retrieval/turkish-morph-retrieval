#!/usr/bin/env python3
"""One-time, auditable policy migration for the 28-family train1250 seed.

Judge outputs and family text are immutable.  Only clean double-pass rows that
were previously flagged solely because confidence was 80--84 are reclassified
from human_review to accept.  Every other warning remains conservative.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from production import load_config
from workflow import Store, atomic_json, contract, read_json

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "train1250"


def main():
    manifest = read_json(RUN / "manifest.json")
    protected = read_json(RUN / "protected.json")
    plan = read_json(RUN / "plan.json")
    old_contract = manifest["contract_sha256"]
    migrated_at = datetime.now(timezone.utc).isoformat()
    store = Store(RUN / "state.sqlite3")
    changed = []
    try:
        for job_id, payload in store.execute(
                "SELECT id,result FROM jobs WHERE status='accepted' ORDER BY id"):
            row = json.loads(payload)
            if (row.get("train_decision") == "human_review"
                    and row.get("review_reason") == "low_confidence_pass"
                    and min(row.get("judge_confidence", {}).values(), default=0) >= 80):
                row["train_decision"] = "accept"
                row["review_reason"] = None
                row["reason"] = "both_judges_pass"
                row.setdefault("events", []).append({
                    "stage": "policy_migration",
                    "from": "train-v24-single-low-relaxed-warnings",
                    "to": "train-v25-balanced-review-routing",
                    "reason": "clean_double_pass_confidence_80_84",
                    "migrated_at": migrated_at,
                })
                store.execute("UPDATE jobs SET result=? WHERE id=?",
                              (json.dumps(row, ensure_ascii=False), job_id))
                changed.append(job_id)
        new_config = load_config()
        new_contract = contract(new_config, protected, plan)
        manifest["config"] = new_config
        manifest["contract_sha256"] = new_contract
        migrations = manifest.setdefault("policy_migrations", [])
        if migrations and migrations[-1].get("to_version") == "train-v25-balanced-review-routing":
            # Safe to finalize the same uncommitted migration after a source-only
            # correction; do not invent a second migration event.
            migrations[-1]["to_contract_sha256"] = new_contract
            migrations[-1]["reclassified_jobs"] = sorted(set(
                migrations[-1].get("reclassified_jobs", []) + changed))
        else:
            migrations.append({
                "from_contract_sha256": old_contract,
                "to_contract_sha256": new_contract,
                "from_version": "train-v24-single-low-relaxed-warnings",
                "to_version": "train-v25-balanced-review-routing",
                "migrated_at": migrated_at,
                "reclassified_jobs": changed,
                "scope": "policy metadata only; family text and judge verdicts unchanged",
            })
        atomic_json(RUN / "manifest.json", manifest)
    finally:
        store.close()
    print(json.dumps({"reclassified": changed, "count": len(changed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
