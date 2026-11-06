"""Create three disjoint manual-review bundles from the first 2,500 train slots."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "train/data/shards"
OUT = ROOT / "train/human_review_2500"
REVIEWERS = ("murat", "burak", "emir")
CAPACITY = {"murat": 323, "burak": 323, "emir": 322}


def slot_id(row):
    return row.get("slot_id") or row.get("family", {}).get("slot_id")


def slot_no(row):
    return int(slot_id(row).rsplit("_", 1)[-1])


def load_rows():
    rows = [
        json.loads(line)
        for path in sorted(SHARDS.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = sorted(rows, key=slot_no)[:2500]
    if len(rows) != 2500 or len({slot_id(row) for row in rows}) != 2500:
        raise ValueError("İlk 2.500 train slotu eksik veya tekrarlı")
    review = [row for row in rows if row.get("train_decision") == "human_review"]
    if len(review) != 968:
        raise ValueError(f"968 human_review bekleniyordu; bulunan: {len(review)}")
    return review


def assign(rows):
    """Round-robin each reason bucket while respecting fixed capacities."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get("review_reason") or "missing_reason"].append(row)
    assigned = {name: [] for name in REVIEWERS}
    cursor = 0
    for reason in sorted(buckets, key=lambda key: (-len(buckets[key]), key)):
        for row in sorted(buckets[reason], key=slot_no):
            for _ in REVIEWERS:
                name = REVIEWERS[cursor % len(REVIEWERS)]
                cursor += 1
                if len(assigned[name]) < CAPACITY[name]:
                    assigned[name].append(row)
                    break
            else:
                raise RuntimeError("Review kapasitesi doldu")
        cursor += 1
    if {name: len(rows) for name, rows in assigned.items()} != CAPACITY:
        raise RuntimeError("Review dağılımı beklenen kapasiteye uymadı")
    return assigned


STYLE = """
body{font:16px/1.55 system-ui;background:#f4f5f7;color:#17212b;margin:0}
header{position:sticky;top:0;background:#102b35;color:white;padding:14px 24px;z-index:5}
main{max-width:1080px;margin:auto;padding:20px}article{background:white;border:1px solid #d8dde2;border-radius:14px;padding:22px;margin:20px 0}
.meta{color:#51616a}.warning{background:#fff1cf;padding:10px;border-radius:8px}.query,.candidate{padding:12px 16px;border-radius:9px;margin:10px 0;background:#edf1f5}
.positive{background:#dcf4e8}.morph_1,.morph_2{background:#ffe9df}.semantic_1{background:#eee8fb}
textarea{box-sizing:border-box;width:100%;min-height:76px;font:15px/1.45 system-ui;padding:9px}.notes{min-height:55px}
label{margin-right:18px}button{padding:9px 14px;font-weight:700}details pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}
"""

SCRIPT = r"""
const reviewer=document.body.dataset.reviewer, hash=document.body.dataset.hash;
const key=`morphtr_train_review:${reviewer}:${hash}`;
const cards=[...document.querySelectorAll('article')];
function collect(){return cards.map(card=>({slot_id:card.dataset.slot,decision:card.querySelector('input[type=radio]:checked')?.value||'',query:card.querySelector('[data-field=query]').value,candidates:Object.fromEntries([...card.querySelectorAll('[data-slot]')].map(x=>[x.dataset.slot,x.value])),notes:card.querySelector('.notes').value}));}
function save(){localStorage.setItem(key,JSON.stringify(collect())); update();}
function update(){const d=collect();document.querySelector('#progress').textContent=`${d.filter(x=>x.decision).length}/${d.length} karar`;}
function restore(){const raw=localStorage.getItem(key);if(!raw)return;for(const x of JSON.parse(raw)){const c=document.querySelector(`article[data-slot="${x.slot_id}"]`);if(!c)continue;const r=c.querySelector(`input[value="${x.decision}"]`);if(r)r.checked=true;c.querySelector('[data-field=query]').value=x.query;for(const [s,v] of Object.entries(x.candidates||{})){const t=c.querySelector(`[data-slot="${s}"]`);if(t)t.value=v;}c.querySelector('.notes').value=x.notes||'';}update();}
document.body.addEventListener('input',save);document.body.addEventListener('change',save);
document.querySelector('#download').onclick=()=>{const out={reviewer,source_sha256:hash,reviewed_at:new Date().toISOString(),decisions:collect()};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download=`${reviewer}_train_human_review.json`;a.click();URL.revokeObjectURL(a.href);};restore();update();
"""


def render(name, rows, source_hash):
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    labels = {"positive": "Positive / gold", "morph_1": "Morph 1", "morph_2": "Morph 2", "semantic_1": "Semantic 1"}
    for i, row in enumerate(sorted(rows, key=slot_no), 1):
        fam = row["family"]
        candidates = "".join(
            f'<div class="candidate {esc(c["slot"])}"><b>{esc(labels[c["slot"]])}</b><textarea data-slot="{esc(c["slot"])}">{esc(c["text"])}</textarea></div>'
            for c in fam["candidates"]
        )
        warnings = ", ".join(row.get("local_quality_warnings") or []) or "yok"
        confidence = json.dumps(row.get("judge_confidence"), ensure_ascii=False)
        cards.append(
            f'<article data-slot="{esc(slot_id(row))}"><h2>{i}. {esc(slot_id(row))} · {esc(fam["target_feature"])}</h2>'
            f'<p class="meta">Neden: <b>{esc(row.get("review_reason") or "eksik")}</b> · Judge confidence: {esc(confidence)}</p>'
            f'<p class="warning">Yerel uyarılar: {esc(warnings)}</p><div class="query"><b>Query</b><textarea data-field="query">{esc(fam["query"])}</textarea></div>{candidates}'
            f'<p><label><input type="radio" name="d{i}" value="pass"> Pass</label><label><input type="radio" name="d{i}" value="fix"> Düzelttim</label><label><input type="radio" name="d{i}" value="reject"> Reject</label></p>'
            f'<textarea class="notes" placeholder="Kısa not / düzeltme gerekçesi"></textarea>'
            f'<details><summary>Tam orijinal kayıt</summary><pre>{esc(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details></article>'
        )
    return f'''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} train review</title><style>{STYLE}</style><body data-reviewer="{name}" data-hash="{source_hash}"><header><b>{name.title()} · {len(rows)} human-review kaydı</b> · <span id="progress"></span> <button id="download">Kararları JSON indir</button></header><main><h1>MorphTR train insan kontrolü</h1><p>Her kayıtta query, positive, iki morfolojik negatif ve bir semantik negatif vardır. Doğruysa Pass; metni düzelttiysen Düzelttim; güvenilir biçimde düzeltilemiyorsa Reject seç. Değişiklikler tarayıcıda otomatik saklanır; düzenli olarak JSON indir.</p>{''.join(cards)}</main><script>{SCRIPT}</script></body></html>'''


def main():
    rows = load_rows()
    source = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    source_hash = sha256(source.encode()).hexdigest()
    assigned = assign(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"scope": "first_2500_train_slots", "human_review_total": len(rows), "source_sha256": source_hash, "assignments": []}
    for name in REVIEWERS:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        subset = sorted(assigned[name], key=slot_no)
        (folder / f"{name}_train_human_review.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset), encoding="utf-8")
        (folder / f"{name}_train_human_review.html").write_text(render(name, subset, source_hash), encoding="utf-8")
        reasons = dict(Counter((row.get("review_reason") or "missing_reason") for row in subset))
        metadata = {"reviewer": name, "count": len(subset), "source_sha256": source_hash, "review_reasons": reasons, "slot_ids": [slot_id(row) for row in subset]}
        (folder / f"{name}_assignment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["assignments"].append(metadata)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {"count": len(assigned[name]), "reasons": dict(Counter((r.get('review_reason') or 'missing_reason') for r in assigned[name]))} for name in REVIEWERS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
