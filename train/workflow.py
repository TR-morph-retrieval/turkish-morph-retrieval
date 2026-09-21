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
    texts, lemmas, all_lemmas, templates, chains, pairs = [], set(), set(), set(), set(), set()
    shared_chains = [f['key'] for f in read_json(HERE/'catalog.json')['features']
                     if f.get('objective') == 'composition']
    chain_lemmas = {key: set() for key in shared_chains}
    for x in rows:
        if len(x['candidates']) != 11:
            raise ValueError('Invalid protected family candidate count')
        texts += [x['query']] + [c['text'] for c in x['candidates']]
        all_lemmas.add(normalized(x['critical_lemma']))
        all_lemmas.update(normalized(c['critical_lemma']) for c in x['candidates']
                          if c.get('critical_lemma'))
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
            'texts': texts, 'forbidden_lemmas': sorted(lemmas),
            'all_test_critical_lemmas': sorted(all_lemmas),
            'forbidden_templates': sorted(templates),
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
                if candidate['slot'] in {'morph_1', 'morph_2'}:
                    if candidate.get('critical_lemma') != positive.get('critical_lemma'):
                        errors.append(f"quality:{candidate['slot']}:lemma_changed_outside_target")
                    if candidate.get('critical_pos') != positive.get('critical_pos'):
                        errors.append(f"quality:{candidate['slot']}:pos_changed_outside_target")
                    base = set(normalized(positive.get('critical_sentence', '')).split())
                    current = set(normalized(candidate.get('critical_sentence', '')).split())
                    base.discard(normalized(positive.get('critical_word', '')))
                    current.discard(normalized(candidate.get('critical_word', '')))
                    # The critical surface form may change, but objects, names and
                    # the rest of the proposition must not quietly drift. A 0.50
                    # token Jaccard floor permits a natural Turkish rewording or
                    # syntactic repair, while still blocking a replaced event/object
                    # such as susam→çörek otu.
                    if (spec['target_feature'] != 'MORPH.CONTEXT_AMBIG'
                            and len(base & current) / max(1, len(base | current)) < 0.50):
                        errors.append(f"quality:{candidate['slot']}:non_target_content_drift")
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
        critical = [('query', item.get('query_critical_word'), item.get('query_critical_lemma'),
                     item.get('query_critical_sentence'), item['query'])]
        critical += [(c.get('slot'), c.get('critical_word'), c.get('critical_lemma'),
                      c.get('critical_sentence'), c['text']) for c in item['candidates']]
        forbidden = set(self.protected['forbidden_lemmas'])
        chain_roots = set(self.protected.get('chain_forbidden_lemmas', {}).get(spec['target_feature'], []))
        for slot, word, lemma, sentence, text in critical:
            if slot == 'semantic_1':
                if not isinstance(sentence, str) or not sentence.strip() or sentence not in sentence_parts(text):
                    errors.append('metadata:semantic_1:critical_sentence_not_in_text')
                if isinstance(sentence, str) and (('?' in sentence) != (spec['target_feature'] == 'Q.PART.SCOPE')):
                    errors.append('plan:question_vs_statement')
                continue
            if not all(isinstance(v, str) and v.strip() for v in [word, lemma, sentence]):
                errors.append(f'metadata:{slot}:missing_critical_annotation'); continue
            if normalized(word) not in normalized(sentence).split() or sentence not in sentence_parts(text):
                errors.append(f'metadata:{slot}:critical_annotation_not_in_text')
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
        words = {w for slot, word, lemma, sentence, text in critical
                 if slot != 'semantic_1' and isinstance(word, str)
                 for w in normalized(word).split()}
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
    sources = {p.name: digest(p.read_text()) for p in [
        HERE/'production.py', HERE/'workflow.py', HERE/'catalog.json',
        HERE/'generation_guides.json', HERE/'lemma_pool.json']}
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
    pool = read_json(HERE/'lemma_pool.json')
    guides = read_json(HERE/'generation_guides.json')
    blocked = set(protected.get('all_test_critical_lemmas', []))
    blocked.update(protected.get('chain_forbidden_lemmas', {}).get(spec['target_feature'], []))
    rng = random.Random(int(digest([spec['slot_id'], spec['target_feature']])[:16], 16))
    choices = []
    for pos in ('VERB', 'NOUN'):
        values = [lemma for lemma in pool[pos] if normalized(lemma) not in blocked]
        rng.shuffle(values)
        choices += [{'lemma': lemma, 'pos': pos} for lemma in values[:5]]
    objective_guides = {
        'composition': {'contract': 'Positive tanımlanan ek zincirinin bütün halkalarını taşır. Morph_1 ve morph_2 aynı fiil kökü, olay ve katılımcılarla zincirin farklı birer halkasını düşürür veya değiştirir; diğer halkalar korunur. Sonuç mutlaka doğal Türkçedir.'},
        'allomorph_invariance': {'contract': 'Query ve positive aynı dilbilgisel işlevin bağlama uygun farklı yüzey biçimlerini taşır. Morph hard aynı lemma ve olayda yüzey biçimini değil işlevi değiştirir; eşdeğer allomorf asla negatif yapılmaz.'},
        'morpheme_sensitivity': {'contract': 'Positive hedef anlamı taşır. Morph hard aynı lemma, olay ve katılımcılarla yalnız hedef morfolojik işlevi değiştirir; sözdizimi değişen biçimin doğal kullanımına gerektiği kadar uyarlanır.'},
    }
    guide = guides.get(spec['target_feature'], objective_guides.get(
        spec['feature'].get('objective'), {
            'contract': 'Positive hedef özelliği taşır. Her morph hard aynı lemma ve olayda yalnız bir planlı morfolojik işlevi değiştirir; morph_change bu tek değişimi doğru yazar.'}))
    return {**spec, 'attempt': attempt, 'preferred_novel_lemmas': choices,
            'feature_generation_guide': guide,
            'required_metadata': 'template_id, domain, register; query_critical_word, query_critical_lemma, query_critical_sentence; each candidate critical_word, critical_lemma, critical_sentence',
            'instructions': 'Plan alanları ve sayıları aynen koru. preferred_novel_lemmas içinden hedefe uygun iki farklı lemma seç: biri query, biri positive+morph adayları için; morph_1/morph_2 positive ile aynı lemma ve POS kullanır. semantic_1 serbesttir. Ek cümleler doğal bağlam olsun. Her kritik cümle tam bir cümle ve metnin parçası olmalı. strict_minimal modunda positive ile morph_1 sadece bir hedef sözcükte farklı, aynı lemma olmalı. Q.PART.SCOPE kritik cümleleri soru; diğerleri bildirim. Test koruma listeleri prompta verilmez; yalnız testte bulunmadığı Python tarafından doğrulanmış lemma seçenekleri verilir.'}


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


def generate_run(folder, client, limit=10, max_calls=30, range_from=None, range_to=None):
    manifest, protected, plan = verify(folder)
    # Train has no human-review gate; two automatic judges are the quality gate.
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
                        previous_family = prior.get('family') or {}
                        previous_lemmas = [previous_family.get('query_critical_lemma')]
                        previous_lemmas += [c.get('critical_lemma')
                                            for c in previous_family.get('candidates', [])]
                        request_spec['do_not_reuse_previous_critical_lemmas'] = sorted({
                            normalized(x) for x in previous_lemmas
                            if isinstance(x, str) and x.strip()})
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
    selected = plan[(range_from - 1 if range_from else 0):(range_to if range_to else len(plan))]
    pending = iter(s for s in selected if store.execute('SELECT status FROM jobs WHERE id=?', (s['slot_id'],))[0][0]
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
    judge_decisions, rejection_reasons, repairs = Counter(), Counter(), 0
    for (payload,) in store.execute("SELECT value FROM events WHERE kind='family_result'"):
        outcome = json.loads(payload)
        rejection_reasons[outcome.get('reason', 'unknown')] += 1
        for event in outcome.get('events', []):
            stage = event.get('stage')
            if stage in {'semantic', 'morphology'} and event.get('verdict', {}).get('decision'):
                judge_decisions[f"{stage}:{event['verdict']['decision']}"] += 1
            if stage == 'repair':
                repairs += 1
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
            'judge_decisions':dict(judge_decisions), 'repairs':repairs,
            'result_reasons':dict(rejection_reasons),
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
    for name in ['prepare','approve','run','status','export','shard-export','shard-sync','shard-status']:
        p=sub.add_parser(name);p.add_argument('--run-id', required=True)
        if name=='prepare':
            p.add_argument('--source', type=Path, required=True);p.add_argument('--size',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
            p.add_argument('--pilot', action='store_true', help='Unreviewed-test pilot; exports are NOT final training data')
        if name=='approve':
            p.add_argument('--source-sha',required=True);p.add_argument('--reviewer',required=True)
            p.add_argument('--confirm-human-review-complete',action='store_true',required=True)
        if name=='run':
            p.add_argument('--limit',type=int,default=10);p.add_argument('--max-calls',type=int,default=30)
            p.add_argument('--from-index',type=int);p.add_argument('--to-index',type=int)
        if name == 'shard-export':
            p.add_argument('--output', required=True); p.add_argument('--producer', required=True)
            p.add_argument('--from-index', type=int, required=True); p.add_argument('--to-index', type=int, required=True)
        if name in {'shard-sync','shard-status'}:
            p.add_argument('--shard-dir', default=str(HERE/'data'/'shards'))
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
            # Final train generation is intentionally automatic; test approval is separate.
            client=OpenRouter(api_key(),load_config()['transport_attempts'])
            # Cross-run local memory must not race another producer's acceptance transaction.
            with run_lock(HERE/'runs'):
                if args.from_index and (args.from_index < 1 or (args.to_index and args.to_index < args.from_index)):
                    parser.error('Geçersiz train aralığı')
                print(json.dumps(generate_run(folder,client,args.limit,args.max_calls,args.from_index,args.to_index),ensure_ascii=False,indent=2))
            return
        if args.cmd in {'shard-export','shard-sync','shard-status'}:
            from shards import export_shard, sync_shards, validate_shards
            shard_dir = Path(getattr(args, 'shard_dir', HERE/'data'/'shards'))
            if args.cmd == 'shard-export':
                value = export_shard(folder, Path(args.output), args.producer, args.from_index, args.to_index)
            elif args.cmd == 'shard-sync':
                value = sync_shards(folder, shard_dir)
            else:
                value = validate_shards(folder, shard_dir)
            print(json.dumps(value, ensure_ascii=False, indent=2)); return
        store=Store(folder/'state.sqlite3')
        try:
            print(json.dumps(report(folder,store) if args.cmd=='status' else {'export':export(folder,store)},ensure_ascii=False,indent=2))
        finally:
            store.close()


if __name__=='__main__':
    main()
