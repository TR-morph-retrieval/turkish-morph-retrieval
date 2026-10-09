#!/usr/bin/env python3
"""Record the v26 review-policy transition without rewriting existing labels."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from production import load_config
from workflow import atomic_json, contract, read_json

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "train1250"


def main():
    manifest = read_json(RUN / "manifest.json")
    protected = read_json(RUN / "protected.json")
    plan = read_json(RUN / "plan.json")
    old_contract = manifest["contract_sha256"]
    with sqlite3.connect(RUN / "state.sqlite3") as con:
        accepted_before = con.execute(
            "SELECT count(*) FROM jobs WHERE status='accepted'").fetchone()[0]
    cfg = load_config()
    new_contract = contract(cfg, protected, plan)
    manifest["config"] = cfg
    manifest["contract_sha256"] = new_contract
    transitions = manifest.setdefault("policy_migrations", [])
    transition = {
        "from_contract_sha256": old_contract,
        "to_contract_sha256": new_contract,
        "from_version": "train-v25-balanced-review-routing",
        "to_version": "train-v26-drift-aware-review",
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "accepted_before_transition": accepted_before,
        "scope": "future decisions only; existing family text, verdicts and labels unchanged",
    }
    if transitions and transitions[-1].get("to_version") == transition["to_version"]:
        transition["from_contract_sha256"] = transitions[-1]["from_contract_sha256"]
        transition["accepted_before_transition"] = transitions[-1]["accepted_before_transition"]
        transitions[-1].update(transition)
    else:
        transitions.append(transition)
    atomic_json(RUN / "manifest.json", manifest)
    print({"accepted_before_transition": accepted_before, "version": cfg["version"]})


if __name__ == "__main__":
    main()
