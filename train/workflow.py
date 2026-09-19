"""Train-only planning, local leakage guard, SQLite resume and CLI. Standard library only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import fcntl
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import threading

from production import (HERE, OpenRouter, TransportError, digest, evaluate,
                        generate_family, load_config, normalized, validate_family)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def source_rows(source):
    source = Path(source).resolve()
    paths = sorted(source.glob('*.jsonl')) if source.is_dir() else [source]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('Protected source JSONL missing')
    rows, hashes = [], {}
    for path in paths:
        content = path.read_text(encoding='utf-8')
        hashes[path.name] = digest(content)
        rows.extend(json.loads(line) for line in content.splitlines() if line.strip())
    return rows, digest(hashes)


def snapshot(source, expected=600):
    rows, sha = source_rows(source)
    if len(rows) != expected or len({x['family_id'] for x in rows}) != expected:
        raise ValueError(f'Expected {expected} unique protected families')
    texts, lemmas, templates, chains, pairs = [], set(), set(), set(), set()
    shared_chains = [f['key'] for f in read_json(HERE/'catalog.json')['features']
                     if f.get('objective') == 'composition']
    chain_lemmas = {key: set() for key in shared_chains}
    for x in rows:
        if len(x['candidates']) != 11:
            raise ValueError('Invalid protected family candidate count')
        texts += [x['query']] + [c['text'] for c in x['candidates']]
        bucket = x['generalization_bucket']
        if bucket == 'lemma_holdout':
            lemmas.add(normalized(x['critical_lemma']))
        if bucket == 'template_holdout':
            templates.add(x['template_id'])
        if bucket == 'composition_holdout':
            chains.add(x['target_feature'])
        if x['target_feature'] in chain_lemmas:
            chain_lemmas[x['target_feature']].add(normalized(x['critical_lemma']))
            for candidate in x['candidates']:
                if candidate.get('critical_lemma'):
                    chain_lemmas[x['target_feature']].add(normalized(candidate['critical_lemma']))
        if 'domain_shift' in x.get('generalization_tags', []):
            pairs.add((x['domain'], x['register']))
    return {'source': str(Path(source).resolve()), 'source_sha256': sha, 'family_count': len(rows),
            'texts': texts, 'forbidden_lemmas': sorted(lemmas), 'forbidden_templates': sorted(templates),
            'lemma_exclusion_scope': 'target_critical_words_only',
            'chain_forbidden_lemmas': {key: sorted(values) for key, values in chain_lemmas.items()},
            'chain_protection_version': 2,
            'forbidden_features': sorted(chains), 'forbidden_domain_register': sorted(pairs),
            'encoding_warning_count': sum(bool(re.search('[ÃÄÅ�]', t)) for t in texts)}


def distributed(distribution, size, seed):
    exact = {k: v * size for k, v in distribution.items()}
    counts = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k]-counts[k]), str(k)))[:size-sum(counts.values())]:
        counts[k] += 1
    values = [k for k, n in counts.items() for _ in range(n)]
    random.Random(seed).shuffle(values)
    return values


def make_plan(protected, size, seed=42):
    if size < 1:
        raise ValueError('Positive plan size required')
    catalog = read_json(HERE / 'catalog.json')
    policy = load_config()['composition_policy']
    shared = {f['key'] for f in catalog['features'] if f.get('objective') == 'composition'}
    eligible = [f for f in catalog['features']
                if f['key'] not in protected['forbidden_features'] or f['key'] in shared]
    missing = shared - set(protected.get('chain_forbidden_lemmas', {}))
    if missing or protected.get('chain_protection_version') != 2:
        raise ValueError(f'Missing root protection for shared chains: {sorted(missing)}; rebuild protection snapshot')
    templates = [t for t in catalog['templates'] if t['id'] not in protected['forbidden_templates']]
    if not eligible or not templates:
        raise ValueError('No train features/templates left after exclusions')
    rng = random.Random(seed)
    eligible.sort(key=lambda f: f['key']); rng.shuffle(eligible)
    modes = distributed({'strict_minimal': .4, 'controlled_diverse': .3, 'natural_retrieval': .3}, size, seed+1)
    qlength = distributed({1: .75, 2: .25}, size, seed+2)
    plength = distributed({1: .3, 2: .3, 3: .3, 4: .1}, size, seed+3)
    chain_slots = distributed({'chain': policy['chain_fraction'], 'single': 1-policy['chain_fraction']}, size, seed+4)
    pools = {'chain': [f for f in eligible if f['key'] in shared],
             'single': [f for f in eligible if f['key'] not in shared]}
    used = Counter()
    plan = []
    for i in range(size):
        group = chain_slots[i]
        feature = pools[group][used[group] % len(pools[group])]
        used[group] += 1
        domain = catalog['domains'][(i // len(eligible) + i) % len(catalog['domains'])]
        registers = [r for r in ['everyday', 'conversational', 'news_report']
                     if [domain, r] not in protected['forbidden_domain_register']]
        if not registers:
            raise ValueError(f'No available register for {domain}')
        plan.append({'slot_id': f'train_{i+1:06d}', 'target_feature': feature['key'],
                     'target_description': feature['meaning_contrast'], 'feature': feature,
                     'domain': domain, 'register': registers[i % len(registers)],
                     'template': templates[(i // len(eligible)+i) % len(templates)],
                     'family_mode': modes[i], 'query_sentence_count': qlength[i],
                     'generalization_policy': 'root_chain_holdout' if group == 'chain' else 'standard',
                     'passage_sentence_count': plength[i]})
    return plan


def sentence_parts(text):
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+', text.strip()) if p.strip()]


class SimilarityIndex:
    """Inverted word index; exact, token-bigram and character-trigram overlap checks.

    Not semantic equivalence detection: paraphrases with disjoint vocabulary may escape.
    """
    def __init__(self, texts=()):
        self.rows, self.postings, self.exact = [], defaultdict(set), set()
        for text in texts:
            self.add(text)

    def add(self, text):
        for part in [text, *sentence_parts(text)]:
            norm = normalized(part)
            if not norm or norm in self.exact:
                continue
            self.exact.add(norm)
            words = norm.split()
            bigrams = set(zip(words, words[1:]))
            chars = {norm[i:i+3] for i in range(max(0, len(norm)-2))}
            idx = len(self.rows)
            self.rows.append((bigrams, chars, norm))
            for word in set(words):
                self.postings[word].add(idx)

    def overlaps(self, text):
        for part in [text, *sentence_parts(text)]:
            norm = normalized(part)
            if norm in self.exact:
                return True
            words = norm.split()
            if len(words) < 4:
                continue
            bigrams = set(zip(words, words[1:]))
            chars = {norm[i:i+3] for i in range(max(0, len(norm)-2))}
            hits = Counter(i for word in set(words) for i in self.postings.get(word, ()))
            for idx, shared in hits.items():
                if shared < 2:
                    continue
                other_b, other_c, other_norm = self.rows[idx]
                if (len(bigrams & other_b)/max(1, len(bigrams | other_b)) >= .65
                    or len(chars & other_c)/max(1, len(chars | other_c)) >= .85
                    or difflib.SequenceMatcher(None, words, other_norm.split(), autojunk=False).ratio() >= .85):
                    return True
        return False


class Guard:
    def __init__(self, protected, accepted=()):
        self.protected = protected
        self.index = SimilarityIndex(protected['texts'])
        for item in accepted:
            self.add(item)

    def add(self, item):
        for text in [item['query'], *[c['text'] for c in item['candidates']]]:
            self.index.add(text)

    def check(self, item, spec):
        errors = []
        if {c['slot'] for c in item['candidates']} != {'positive','morph_1','morph_2','semantic_1'}:
            errors.append('plan:four_candidates_required')
        if item.get('target_feature') != spec['target_feature'] or item.get('target_description') != spec['target_description']:
            errors.append('plan:target_changed')
        if item.get('template_id') != spec['template']['id']:
            errors.append('plan:template_changed')
        if item.get('domain') != spec['domain'] or item.get('register') != spec['register']:
            errors.append('plan:domain_or_register_changed')
        texts = [item['query']] + [c['text'] for c in item['candidates']]
        positive = next((c for c in item['candidates'] if c['slot'] == 'positive'), {})
        if len(sentence_parts(item.get('query', ''))) != spec['query_sentence_count']:
            errors.append('quality:query_sentence_count')
        if len(item.get('context_sentences', [])) != spec['passage_sentence_count'] - 1:
            errors.append('quality:shared_context_sentence_count')
        query_sentence = normalized(item.get('query_critical_sentence', item['query']))
        if query_sentence and any(query_sentence == normalized(s) for s in sentence_parts(positive.get('text', ''))):
            errors.append('quality:query_sentence_copied_into_positive')
        if positive.get('critical_sentence'):
            context = lambda c: [normalized(s) for s in sentence_parts(c['text'])
                                 if s != c.get('critical_sentence')]
            for candidate in item['candidates']:
                if candidate['slot'] in {'morph_1', 'morph_2'} and context(candidate) != context(positive):
                    errors.append(f"quality:{candidate['slot']}:context_changed")
                if candidate['slot'] == 'morph_1' and spec['family_mode'] == 'strict_minimal':
                    if candidate.get('critical_lemma') != positive.get('critical_lemma'):
                        errors.append('quality:morph_1:strict_lemma_changed')
                    def skeleton(c):
                        return [ '__TARGET__' if token == normalized(c.get('critical_word', '')) else token
                                 for token in normalized(c.get('critical_sentence', '')).split()]
                    if skeleton(candidate) != skeleton(positive):
                        errors.append('quality:morph_1:strict_non_target_edit')
        if any(self.index.overlaps(t) for t in texts):
            errors.append('leakage:protected_or_accepted_text_overlap')
        critical = [(item.get('query_critical_word'), item.get('query_critical_lemma'), item.get('query_critical_sentence'), item['query'])]
        critical += [(c.get('critical_word'), c.get('critical_lemma'), c.get('critical_sentence'), c['text']) for c in item['candidates']]
        forbidden = set(self.protected['forbidden_lemmas'])
        chain_roots = set(self.protected.get('chain_forbidden_lemmas', {}).get(spec['target_feature'], []))
        for word, lemma, sentence, text in critical:
            if not all(isinstance(v, str) and v.strip() for v in [word, lemma, sentence]):
                errors.append('metadata:missing_critical_annotation'); continue
            if normalized(word) not in normalized(sentence).split() or sentence not in sentence_parts(text):
                errors.append('metadata:critical_annotation_not_in_text')
            if normalized(lemma) in forbidden:
                errors.append('leakage:heldout_lemma')
            if normalized(lemma) in chain_roots:
                errors.append('leakage:heldout_root_chain_pair')
            if any(normalized(word) == root or (len(root) >= 4 and normalized(word).startswith(root))
                   for root in chain_roots):
                errors.append('leakage:heldout_root_chain_pair_surface_match')
            if ('?' in sentence) != (spec['target_feature'] == 'Q.PART.SCOPE'):
                errors.append('plan:question_vs_statement')
        # Target-lemma holdout: incidental context words are NOT globally banned.
        words = {w for word, lemma, sentence, text in critical if isinstance(word, str) for w in normalized(word).split()}
        if any(w == lemma or (len(lemma) >= 4 and w.startswith(lemma)) for w in words for lemma in forbidden):
            errors.append('leakage:heldout_lemma_surface_match')
        return sorted(set(errors))


@contextmanager
def run_lock(folder):
    with (folder / '.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this train run') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript('''PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,spec TEXT,status TEXT,attempt INTEGER,draft TEXT,provenance TEXT,result TEXT);
        CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,value TEXT,provenance TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,job TEXT,kind TEXT,value TEXT,created TEXT);
        ''')

    def execute(self, sql, values=()):
        with self.lock:
            cursor = self.db.execute(sql, values)
            result = cursor.fetchall()
            self.db.commit()
            return result

    def event(self, job, kind, value):
        self.execute('INSERT INTO events(job,kind,value,created) VALUES(?,?,?,?)', (job, kind, json.dumps(value, ensure_ascii=False), now()))

    def accepted(self):
        return [json.loads(row[0])['family'] for row in self.execute("SELECT result FROM jobs WHERE status='accepted' ORDER BY id")]

    def close(self):
        self.db.close()


class CachedClient:
    def __init__(self, store, client, max_calls, job=None, budget=None):
        self.store, self.client = store, client
        self.budget = budget if budget is not None else {'remaining': max_calls, 'lock': threading.Lock()}
        self.lock = self.budget['lock']
        self.job = job

    @property
    def remaining(self):
        with self.lock:
            return self.budget['remaining']

    def call(self, settings, prompt):
        key = digest([settings, prompt])
        cached = self.store.execute('SELECT value,provenance FROM cache WHERE key=?', (key,))
        if cached:
            return json.loads(cached[0][0]), {**json.loads(cached[0][1]), 'cache_hit': True}
        with self.lock:
            if self.budget['remaining'] <= 0:
                raise TransportError('Per-invocation logical call cap reached')
            self.budget['remaining'] -= 1
        self.store.event(self.job, 'request_started', {'key': key, 'model': settings['model']})
        try:
            value, provenance = self.client.call(settings, prompt)
        except TransportError as exc:
            self.store.event(self.job, 'request_failed', {'key':key, 'attempts':getattr(exc,'attempts',[])})
            raise
        self.store.execute('INSERT OR REPLACE INTO cache VALUES(?,?,?)', (key, json.dumps(value, ensure_ascii=False), json.dumps(provenance, ensure_ascii=False)))
        self.store.event(self.job, 'request_finished', {'key': key, 'provenance': provenance})
        return value, provenance


def contract(cfg, protected, plan):
    # Exact source identity ensures different code/config cannot silently resume the run.
    sources = {p.name: digest(p.read_text()) for p in [HERE/'production.py', HERE/'workflow.py', HERE/'catalog.json']}
    return digest({'sources': sources, 'config': cfg, 'protected': protected, 'plan': plan})


def prepare(folder, source, size, seed, pilot=False):
    protected = snapshot(source)
    plan = make_plan(protected, size, seed)
    cfg = load_config()
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {'version': 'train-run-v1', 'created': now(), 'reviewed': False,
                'contract_sha256': contract(cfg, protected, plan), 'config': cfg,
                'purpose': 'pilot_only' if pilot else 'final_train',
                'eligible_for_final_train': not pilot}
    atomic_json(folder/'protected.json', protected)
    atomic_json(folder/'plan.json', plan)
    atomic_json(folder/'manifest.json', manifest)
    store = Store(folder/'state.sqlite3')
    try:
        for spec in plan:
            store.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?)', (spec['slot_id'], json.dumps(spec,ensure_ascii=False), 'pending', 1, None, None, None))
    finally:
        store.close()


def verify(folder):
    manifest, protected, plan = [read_json(folder/name) for name in ['manifest.json','protected.json','plan.json']]
    if contract(load_config(), protected, plan) != manifest['contract_sha256']:
        raise ValueError('Train source/config/plan contract changed; use a new run')
    if source_rows(protected['source'])[1] != protected['source_sha256']:
        raise ValueError('Protected test changed; rebuild snapshot and train plan')
    return manifest, protected, plan


def approve(folder, source_sha, reviewer):
    manifest, protected, _ = verify(folder)
    if source_sha != protected['source_sha256'] or not reviewer.strip():
        raise ValueError('Exact source checksum and reviewer name required')
    if protected['encoding_warning_count']:
        raise ValueError('Protected test still has encoding errors; cannot approve')
    manifest.update(reviewed=True, reviewed_by=reviewer, reviewed_at=now())
    atomic_json(folder/'manifest.json', manifest)


def generation_spec(spec, protected, attempt):
    shared = {f['key'] for f in read_json(HERE/'catalog.json')['features']
              if f.get('objective') == 'composition'}
    return {**spec, 'attempt': attempt,
            'forbidden_lemmas': protected['forbidden_lemmas'],
            'forbidden_root_chain_pairs': protected['chain_forbidden_lemmas'],
            'forbidden_features': sorted(set(protected['forbidden_features']) - shared),
            'shared_chain_features': sorted(shared),
            'forbidden_templates':protected['forbidden_templates'],
            'required_metadata': 'template_id, domain, register; query_critical_word, query_critical_lemma, query_critical_sentence; each candidate critical_word, critical_lemma, critical_sentence',
            'instructions': 'Plan alanları ve sayıları aynen koru. Ek cümleler doğal bağlam olsun. forbidden_root_chain_pairs eşlemesindeki zinciri belirtilen kökle query, aday veya bağlamda kullanma. Aynı kök başka zincirlerle, aynı zincir başka köklerle serbesttir; ayrı forbidden_lemmas yasağı saklıdır. Her kritik cümle tam bir cümle ve metnin parçası olmalı. strict_minimal modunda positive ile morph_1 sadece bir hedef sözcükte farklı, aynı lemma olmalı. Q.PART.SCOPE kritik cümleleri soru; diğerleri bildirim.'}


def other_run_memory(folder):
    items = []
    for dbpath in folder.parent.glob('*/state.sqlite3'):
        if dbpath.parent == folder:
            continue
        connection = sqlite3.connect(dbpath.as_uri()+'?mode=ro', uri=True)
        try:
            items += [json.loads(row[0])['family'] for row in connection.execute("SELECT result FROM jobs WHERE status='accepted'")]
        finally:
            connection.close()
    return items


def generate_run(folder, client, limit=10, max_calls=30):
    manifest, protected, plan = verify(folder)
    if not manifest['reviewed'] and manifest.get('purpose') != 'pilot_only':
        raise ValueError('Human-reviewed frozen test approval required; no paid requests sent')
    cfg = manifest['config']
    store = Store(folder/'state.sqlite3')
    budget = {'remaining': max_calls, 'lock': threading.Lock()}
    guard = Guard(protected, [*store.accepted(), *other_run_memory(folder)])
    guard_lock = threading.RLock()
    def checked(item, spec):
        with guard_lock:
            return guard.check(item, spec)

    def process(spec):
        sid = spec['slot_id']
        cached = CachedClient(store, client, max_calls, job=sid, budget=budget)
        while True:
            row = store.execute('SELECT status,attempt,draft,provenance,result FROM jobs WHERE id=?', (sid,))[0]
            status, attempt, draft, provenance, previous = row
            if status in {'accepted','exhausted'}:
                break
            verify(folder)
            try:
                if draft is None:
                    request_spec = generation_spec(spec, protected, attempt)
                    if previous:
                        prior = json.loads(previous)
                        request_spec['previous_errors'] = [e for stage in prior.get('events', []) for e in stage.get('errors', [])]
                        request_spec['previous_errors'] += [
                            f"{stage.get('stage')}: {finding['reason']}"
                            for stage in prior.get('events', [])
                            for finding in stage.get('findings', [])
                            if isinstance(finding, dict) and isinstance(finding.get('reason'), str)]
                        prior_text = ' '.join([prior.get('family', {}).get('query_critical_word', ''), *[c.get('critical_word', '') for c in prior.get('family', {}).get('candidates', [])]])
                        request_spec['avoid_previous_forbidden_words'] = sorted({w for w in normalized(prior_text).split() for lemma in protected['forbidden_lemmas'] if w == lemma or (len(lemma) >= 4 and w.startswith(lemma))})
                    item, prov = generate_family(cached, cfg, request_spec)
                    store.execute('UPDATE jobs SET draft=?,provenance=? WHERE id=?', (json.dumps(item,ensure_ascii=False), json.dumps(prov), sid))
                else:
                    item, prov = json.loads(draft), json.loads(provenance)
                if isinstance(item, dict):
                    item['train_constraints'] = {k:v for k,v in generation_spec(spec,protected,attempt).items()
                                                 if k not in {'slot_id','attempt','required_metadata','instructions'}}
                outcome = evaluate(item, cached, cfg, lambda x: checked(x, spec))
            except TransportError as exc:
                store.event(sid, 'deferred_transport', {'reason':str(exc)})
                return 'deferred'
            outcome.update(slot_id=sid, generation_attempt=attempt, generator=prov, generation_spec=spec)
            store.event(sid, 'family_result', outcome)
            if outcome['status'] == 'deferred_transport':
                store.execute('UPDATE jobs SET result=? WHERE id=?',(json.dumps(outcome,ensure_ascii=False),sid))
                return 'deferred'
            if outcome['status'] == 'accepted':
                verify(folder)
                with guard_lock:
                    # Atomic recheck + commit prevents concurrent near-duplicate acceptance.
                    conflicts = guard.check(outcome['family'], spec)
                    if not conflicts:
                        outcome['family']['family_id'] = sid
                        store.execute("UPDATE jobs SET status='accepted',result=? WHERE id=?", (json.dumps(outcome,ensure_ascii=False),sid))
                        guard.add(outcome['family'])
                        return 'accepted'
                    outcome.update(status='rejected', reason='concurrent_duplicate_or_guard_failure')
                    outcome['events'].append({'stage': 'local_validation', 'errors': conflicts})
                    store.event(sid, 'commit_rejected', outcome)
            if attempt >= cfg['max_generation_attempts']:
                store.execute("UPDATE jobs SET status='exhausted',result=? WHERE id=?", (json.dumps(outcome,ensure_ascii=False),sid))
                break
            store.execute("UPDATE jobs SET attempt=attempt+1,draft=NULL,provenance=NULL,result=? WHERE id=?",(json.dumps(outcome,ensure_ascii=False),sid))
            if cached.remaining <= 0:
                return 'deferred'
        return 'finished'

    accepted_now = 0
    pending = iter(s for s in plan if store.execute('SELECT status FROM jobs WHERE id=?', (s['slot_id'],))[0][0]
                   not in {'accepted', 'exhausted'})
    try:
        with ThreadPoolExecutor(max_workers=cfg['family_workers']) as pool:
            while accepted_now < limit and budget['remaining'] > 0:
                # At most the remaining acceptance allowance is in flight.
                batch = []
                for _ in range(min(cfg['family_workers'], limit - accepted_now)):
                    spec = next(pending, None)
                    if spec is None:
                        break
                    batch.append(spec)
                if not batch:
                    break
                outcomes = list(pool.map(process, batch))
                accepted_now += outcomes.count('accepted')
                if 'deferred' in outcomes:
                    break
        return report(folder, store)
    finally:
        store.close()


def report(folder, store):
    plan = read_json(folder/'plan.json')
    counts = dict(store.execute('SELECT status,COUNT(*) FROM jobs GROUP BY status'))
    accepted = store.accepted()
    request_events = store.execute("SELECT kind,value FROM events WHERE kind IN ('request_finished','request_failed')")
    known_cost, missing_cost, attempts = 0.0, 0, 0
    for kind, payload in request_events:
        obj = json.loads(payload)
        history = obj.get('provenance',{}).get('attempts',[]) if kind=='request_finished' else obj.get('attempts',[])
        for attempt in history:
            attempts += 1
            cost = (attempt.get('usage') or {}).get('cost')
            if isinstance(cost, (int,float)):
                known_cost += cost
            else:
                missing_cost += 1
    return {'planned':len(plan), 'statuses':counts, 'accepted':len(accepted),
            'plan_features':dict(Counter(x['target_feature'] for x in plan)),
            'plan_modes':dict(Counter(x['family_mode'] for x in plan)),
            'accepted_features':dict(Counter(x['target_feature'] for x in accepted)),
            'known_cost_usd':round(known_cost,6), 'attempts_without_cost':missing_cost,
            'recorded_provider_attempts':attempts,
            'unresolved_requests': len(store.execute("SELECT id FROM events WHERE kind='request_started'"))-len(request_events)}


def export(folder, store):
    verify(folder)
    path = folder/'accepted.jsonl'
    tmp = path.with_suffix('.jsonl.tmp')
    with tmp.open('w',encoding='utf-8') as handle:
        for (payload,) in store.execute("SELECT result FROM jobs WHERE status='accepted' ORDER BY id"):
            row = json.loads(payload)
            row['purpose'] = read_json(folder/'manifest.json').get('purpose', 'final_train')
            row['eligible_for_final_train'] = row['purpose'] != 'pilot_only'
            family = row['family']
            # Deterministic training view + full provenance in ONE JSONL, no extra duplicates.
            row['training'] = {'query':family['query'],
                               'positive':next(c['text'] for c in family['candidates'] if c['slot']=='positive'),
                               'negatives':[c['text'] for c in family['candidates'] if c['slot']!='positive']}
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)
    atomic_json(folder/'report.json', report(folder,store))
    return str(path)


def api_key():
    """Read only the shared credential, never any TEST_* configuration."""
    value = os.environ.get('OPENROUTER_API_KEY')
    if value:
        return value
    env_file = HERE.parent/'.env'
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            match = re.fullmatch(r'\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*?)\s*',line)
            if match:
                return match[1].strip('\"\'')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ['prepare','approve','run','status','export']:
        p=sub.add_parser(name);p.add_argument('--run-id', required=True)
        if name=='prepare':
            p.add_argument('--source', type=Path, required=True);p.add_argument('--size',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
            p.add_argument('--pilot', action='store_true', help='Unreviewed-test pilot; exports are NOT final training data')
        if name=='approve':
            p.add_argument('--source-sha',required=True);p.add_argument('--reviewer',required=True)
            p.add_argument('--confirm-human-review-complete',action='store_true',required=True)
        if name=='run':
            p.add_argument('--limit',type=int,default=10);p.add_argument('--max-calls',type=int,default=30)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id):
        parser.error('Invalid run id')
    folder=(HERE/'runs'/args.run_id).resolve()
    if not folder.is_relative_to(HERE.resolve()):
        parser.error('Output must remain under train/')
    if args.cmd=='prepare':
        prepare(folder,args.source,args.size,args.seed,args.pilot)
        print(json.dumps({'run':str(folder),'reviewed':False,'source_sha256':read_json(folder/'protected.json')['source_sha256']},indent=2));return
    if not folder.is_dir():
        parser.error('Run missing; prepare first')
    with run_lock(folder):
        if args.cmd=='approve':
            approve(folder,args.source_sha,args.reviewer);print('Reviewed source approved');return
        if args.cmd=='run':
            if args.limit < 1 or args.max_calls < 1:
                parser.error('Positive limit/max-calls required')
            verify(folder)
            manifest = read_json(folder/'manifest.json')
            if not manifest['reviewed'] and manifest.get('purpose') != 'pilot_only':
                raise ValueError('Reviewed test approval required; no API call made')
            client=OpenRouter(api_key(),load_config()['transport_attempts'])
            # Cross-run local memory must not race another producer's acceptance transaction.
            with run_lock(HERE/'runs'):
                print(json.dumps(generate_run(folder,client,args.limit,args.max_calls),ensure_ascii=False,indent=2))
            return
        store=Store(folder/'state.sqlite3')
        try:
            print(json.dumps(report(folder,store) if args.cmd=='status' else {'export':export(folder,store)},ensure_ascii=False,indent=2))
        finally:
            store.close()


if __name__=='__main__':
    main()
