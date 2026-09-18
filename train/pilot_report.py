"""Offline pilot-only bundle and escaped HTML; never modifies production records."""
import argparse
import html
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent / 'runs'


def bundle(names, output):
    records, costs = [], {}
    for name in names:
        folder = ROOT / name
        db = sqlite3.connect((folder / 'state.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)
        try:
            for (payload,) in db.execute("SELECT result FROM jobs WHERE status='accepted' ORDER BY id"):
                row = json.loads(payload)
                records.append({'review_id': f'{name}:{row["slot_id"]}', 'source_run': name,
                                'purpose': 'pilot_only', 'eligible_for_final_train': False,
                                'record': row})
            for kind, payload in db.execute("SELECT kind,value FROM events WHERE kind IN ('request_finished','request_failed')"):
                event = json.loads(payload)
                prov = event.get('provenance', {})
                attempts = prov.get('attempts', []) if kind == 'request_finished' else event.get('attempts', [])
                for a in attempts:
                    model = a.get('model') or prov.get('requested_model') or 'unknown'
                    t = costs.setdefault(model, {'calls': 0, 'input_tokens': 0, 'output_tokens': 0,
                                                'cost_usd': 0, 'missing_cost': 0, 'providers': {}})
                    usage = a.get('usage') or {}
                    t['calls'] += 1
                    t['input_tokens'] += usage.get('prompt_tokens', 0)
                    t['output_tokens'] += usage.get('completion_tokens', 0)
                    cost = usage.get('cost')
                    if isinstance(cost, (int, float)):
                        t['cost_usd'] += cost
                    else:
                        t['missing_cost'] += 1
                    provider = a.get('provider', 'unknown')
                    t['providers'][provider] = t['providers'].get(provider, 0) + 1
        finally:
            db.close()
    output.mkdir(parents=True, exist_ok=True)
    data = {'purpose': 'pilot_only', 'records': records, 'costs': costs}
    (output / 'data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    audit_input = [{'review_id': r['review_id'], 'family': r['record']['family']} for r in records]
    (output / 'audit_input.json').write_text(json.dumps(audit_input, ensure_ascii=False, indent=2), encoding='utf-8')
    return data


def render(output):
    data = json.loads((output / 'data.json').read_text())
    audit_path = output / 'sol_review.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {'reviews': [
        {'review_id': r['review_id'], 'decision': 'not_audited',
         'summary': 'Ek Sol kontrolü yapılmadı; otomatik kabul bağımsız kalite garantisi değildir.'}
        for r in data['records']]}
    reviews = {r['review_id']: r for r in audit['reviews']}
    expected = {r['review_id'] for r in data['records']}
    if set(reviews) != expected or len(audit['reviews']) != len(expected):
        raise ValueError('Audit must cover every unique pilot ID exactly once')
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    labels = {'positive': 'Positive / gold', 'morph_1': 'Morfolojik negatif 1',
              'morph_2': 'Morfolojik negatif 2', 'semantic_1': 'Semantik negatif'}
    for i, row in enumerate(data['records'], 1):
        family = row['record']['family']; review = reviews[row['review_id']]
        candidates = ''.join(f'<div class="candidate {esc(c["slot"])}"><b>{esc(labels.get(c["slot"], c["slot"]))}</b><p>{esc(c["text"])}</p></div>' for c in family['candidates'])
        issues = ''.join(f'<li>{esc(x)}</li>' for x in review.get('issues', []))
        cards.append(f'<article id="f{i}"><h2>{i}. {esc(family["target_feature"])}</h2><small>{esc(row["review_id"])}</small><div class="query"><b>Query</b><p>{esc(family["query"])}</p></div>{candidates}<aside><b>Sol kontrolü: {esc(review["decision"])}</b><p>{esc(review["summary"])}</p><ul>{issues}</ul></aside><details><summary>Orijinal generator ve judge kayıtları</summary><pre>{esc(json.dumps(row["record"], ensure_ascii=False, indent=2))}</pre></details></article>')
    total = sum(t['cost_usd'] for t in data['costs'].values())
    doc = f'''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Train pilot — {len(cards)} örnek</title><style>body{{font:17px/1.65 system-ui;background:#f5f6f8;color:#19212c;max-width:1050px;margin:auto;padding:24px}}article{{background:white;margin:25px 0;padding:26px;border-radius:16px;border:1px solid #ddd}}h1,h2{{line-height:1.3}}.query,.candidate,aside{{padding:15px 20px;border-radius:10px;margin:12px 0;background:#edf0f5}}.positive{{background:#e0f4e9}}.morph_1,.morph_2{{background:#fff0e9}}.semantic_1{{background:#eee9fa}}aside{{background:#fff8dc}}p{{margin:6px 0}}small{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}nav a{{display:inline-block;margin:4px;padding:6px 12px;background:white;color:#164675;border-radius:8px}}@media print{{details,nav{{display:none}}article{{break-inside:avoid}}}}</style><h1>Train pilot — {len(cards)} family</h1><p>Her family: 1 query, 1 positive, 2 morfolojik negatif, 1 semantik negatif. Üretici Gemini 3.8 Flash; judge'lar DeepSeek V4 Flash 0731 ve GLM 5.3 Flash. Yalnız pilot: final train'e uygun olarak işaretlenmemiştir.</p><p>Sol değerlendirmesi ayrı bir model kontrolüdür; insan onayı veya doğruluk garantisi değildir. Orijinal kayıtlar değiştirilmemiştir. Seçilen üç pilot koşunun kaydedilen toplam ücreti: ${total:.6f}; önceki diğer başarısız deneyler dahil değildir.</p><nav>{''.join(f'<a href="#f{i}">{i}</a>' for i in range(1,len(cards)+1))}</nav>{''.join(cards)}</html>'''
    doc = doc.replace('Seçilen üç pilot koşunun', 'Seçilen pilot koşuların')
    (output / 'pilot.html').write_text(doc, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['bundle', 'html'])
    parser.add_argument('--runs', nargs='+', default=['pilot5_v4'])
    parser.add_argument('--output', default='pilot_review')
    args = parser.parse_args()
    if Path(args.output).name != args.output or args.output in {'.', '..'}:
        parser.error('Output must be one folder name')
    output = ROOT / args.output
    if args.action == 'bundle':
        print(len(bundle(args.runs, output)['records']))
    else:
        render(output); print(output / 'pilot.html')
