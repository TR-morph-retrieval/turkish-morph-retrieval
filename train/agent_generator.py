#!/usr/bin/env python3
"""
Agent Generator Bridge with Full OpenRouter Judges:
- Generator: Antigravity Session (Gemini 3.8 Flash) [Ücretsiz]
- Judges: OpenRouter üzerinden gpt-5.6-luna (Semantic) + glm-5.3-flash (Morphology)
- Policy: policy() ve Guard.check() doğrulaması
"""
import sys
import json
import time
from pathlib import Path
from copy import deepcopy

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from workflow import verify, generation_spec, Guard, Store, read_json, now, api_key
from production import (
    validate_family, materialize, canonicalize_query_annotation,
    canonicalize_candidate_annotations, FACT_KEYS, OpenRouter,
    judge_prompt, policy, load_config
)

def get_pending_spec(folder_path, slot_id=None):
    folder = Path(folder_path)
    manifest, protected, plan = verify(folder)
    store = Store(folder / "state.sqlite3")
    query = "SELECT id, spec FROM jobs WHERE status='pending' ORDER BY id"
    if slot_id:
        query = f"SELECT id, spec FROM jobs WHERE id='{slot_id}'"
    rows = store.execute(query)
    if not rows:
        return None
    target_id, spec_json = rows[0]
    spec = json.loads(spec_json)
    gspec = generation_spec(spec, protected, 1)
    return {
        "slot_id": target_id,
        "spec": spec,
        "generation_spec": gspec,
        "protected_summary": {
            "all_test_critical_lemmas_count": len(protected.get("all_test_critical_lemmas", [])),
            "forbidden_lemmas_sample": protected.get("forbidden_lemmas", [])[:10],
        }
    }

def judge_and_commit_family(folder_path, raw_family):
    start_time = time.time()
    folder = Path(folder_path)
    manifest, protected, plan = verify(folder)
    store = Store(folder / "state.sqlite3")
    guard = Guard(protected, store.accepted())
    cfg = load_config()

    slot_id = raw_family["slot_id"]
    spec_rows = store.execute(f"SELECT spec FROM jobs WHERE id='{slot_id}'")
    if not spec_rows:
        return {"status": "error", "message": f"Slot {slot_id} not found"}
    spec = json.loads(spec_rows[0][0])

    # Metadata doldurma
    item = deepcopy(raw_family)
    item["target_feature"] = spec["target_feature"]
    item["target_description"] = spec["target_description"]
    item["template_id"] = spec["template"]["id"]
    item["domain"] = spec["domain"]
    item["register"] = spec["register"]
    
    frame = item.get("event_frame", {})
    item["event_frame"] = {
        key: (str(frame.get(key)).strip() if isinstance(frame, dict) and frame.get(key) is not None and str(frame.get(key)).strip() else "unspecified")
        for key in FACT_KEYS
    }

    # Materialize (context_sentences -> candidate texts)
    mat_item = materialize(item)
    canon_item = canonicalize_candidate_annotations(canonicalize_query_annotation(mat_item))

    # 1. Yerel Yapısal ve Guard Doğrulaması
    val_errors = validate_family(canon_item)
    if val_errors:
        return {"status": "validation_failed", "errors": val_errors}

    guard_errors = guard.check(canon_item, spec)
    if guard_errors:
        return {"status": "guard_failed", "errors": guard_errors}

    # 2. OpenRouter LLM-as-a-Judge Hakemleri Çağrısı
    client = OpenRouter(api_key(), attempts=cfg.get("transport_attempts", 3))
    order = canon_item['candidates']
    valid_ids = {f'c{i}' for i in range(len(order))}

    reports = {}
    provenances = {}
    judge_events = []
    total_cost = 0.0

    for kind in ('semantic', 'morphology'):
        judge_settings = cfg['judges'][kind]
        prompt = judge_prompt(canon_item, kind, order)
        verdict, prov = client.call(judge_settings, prompt)
        reports[kind] = verdict
        provenances[kind] = prov

        cost = prov.get('attempts', [{}])[0].get('usage', {}).get('cost', 0.0)
        total_cost += cost

        judge_events.append({
            'stage': kind,
            'round': 0,
            'verdict': verdict,
            'provenance': prov
        })

    # 3. Policy Kontrolü
    threshold = cfg.get('confidence_threshold', 80)
    pass_threshold = cfg.get('pass_confidence_threshold', 85)
    decision = policy(reports, valid_ids, threshold=threshold, pass_threshold=pass_threshold)

    elapsed = time.time() - start_time

    if decision.get('action') != 'accept':
        return {
            "status": "judges_rejected",
            "decision": decision,
            "reports": reports,
            "cost": total_cost,
            "elapsed_seconds": elapsed
        }

    # 4. Başarılı Kayıt
    judge_events.append({
        'stage': 'policy',
        'round': 0,
        'action': 'accept'
    })

    gen_prov = {
        "model": "google/gemini-3.8-flash",
        "source": "antigravity-session",
        "timestamp": now()
    }

    result_data = {
        "status": "accepted",
        "reason": "both_judges_pass",
        "family": canon_item,
        "repairs": 0,
        "config_sha256": manifest["contract_sha256"],
        "events": judge_events,
        "slot_id": slot_id,
        "generation_attempt": 1,
        "generator": gen_prov,
        "generation_spec": spec,
        "pilot_review_id": f"{folder.name}:{slot_id}",
        "source_run": folder.name,
        "purpose": manifest.get("purpose", "pilot_only"),
        "eligible_for_final_train": manifest.get("eligible_for_final_train", False)
    }

    store.execute(
        "UPDATE jobs SET status=?, attempt=?, draft=?, provenance=?, result=? WHERE id=?",
        ("accepted", 1, json.dumps(canon_item, ensure_ascii=False), json.dumps({"generator": gen_prov, "judges": provenances}, ensure_ascii=False), json.dumps(result_data, ensure_ascii=False), slot_id)
    )
    store.event(slot_id, "accepted", {"reason": "both_judges_pass", "cost": total_cost})

    return {
        "status": "success",
        "slot_id": slot_id,
        "family": canon_item,
        "judges": reports,
        "cost": total_cost,
        "elapsed_seconds": elapsed
    }

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pending":
        folder = sys.argv[2] if len(sys.argv) > 2 else "train/runs/pilot_review_v24_20260923_a"
        slot = sys.argv[3] if len(sys.argv) > 3 else None
        print(json.dumps(get_pending_spec(folder, slot), ensure_ascii=False, indent=2))
