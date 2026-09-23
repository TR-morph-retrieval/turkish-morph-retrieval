"""Offline end-to-end train tests. Synthetic fixtures are never training data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from production import TransportError
from production import FACT_KEYS, load_config
from test_production import add_checks
from workflow import (Guard, SimilarityIndex, Store, approve, contract, export, generate_run,
                      make_plan, prepare, read_json, report, run_lock, snapshot, verify, CachedClient)
from shards import export_shard, validate_shards, sync_shards


SPEC = {'slot_id':'train_000001','target_feature':'NEG','target_description':'Eylem gerçekleşmez',
        'feature':{'key':'NEG'},'domain':'daily_life','register':'everyday',
        'template':{'id':'event_report','description':'doğal olay bildirimi'},
        'family_mode':'strict_minimal','query_sentence_count':1,'passage_sentence_count':1}


def fixture():
    item = dict(query='Suna dün havuzda suya girmedi.', query_critical_word='girmedi',
                event_frame={k:'unspecified' for k in FACT_KEYS}, context_sentences=[], critical_position=0,
                query_critical_lemma='gir', query_critical_sentence='Suna dün havuzda suya girmedi.',
                target_feature='NEG',target_description='Eylem gerçekleşmez',template_id='event_report',
                domain='daily_life',register='everyday',candidates=[])
    for slot, text, word in [
        ('positive','Suna dün havuzda hiç yüzmedi.','yüzmedi'),
        ('morph_1','Suna dün havuzda hiç yüzdü.','yüzdü'),
        ('morph_2','Suna dün havuzda hiç yüzmeyecek.','yüzmeyecek'),
        ('semantic_1','Yelda dün gölette hiç yüzmedi.','yüzmedi')]:
        item['candidates'].append(dict(slot=slot,text=text,critical_sentence=text,critical_word=word,critical_lemma='yüz',critical_pos='VERB'))
        if slot.startswith('morph_'):
            item['candidates'][-1]['critical_pos'] = 'VERB'
            item['candidates'][-1]['morph_change'] = {'feature':'NEG', 'from':'negative', 'to':'affirmative'}
    return item


class Client:
    def __init__(self, fail_once=False):
        self.calls=0;self.fail_once=fail_once

    def call(self, settings, prompt):
        self.calls+=1
        if 'Sabit çekirdek:\n' in prompt:
            value={'candidates': [c for c in fixture()['candidates'] if c['slot'] != 'positive']}
        elif 'Train slotu:\n' in prompt:
            value=fixture()
            # Hybrid production first creates only query/positive; preserve the
            # fixture's shared metadata while matching that response contract.
            if 'ilk aşamasısın' in prompt:
                value['positive'] = next(c for c in value['candidates'] if c['slot'] == 'positive')
                value.pop('candidates')
        else:
            if not settings['model'].startswith('z-ai/') and self.fail_once:
                self.fail_once=False
                raise TransportError('temporary fixture failure')
            data=json.loads(prompt.split('Veri:\n')[1])
            cid=next(c['candidate_id'] for c in data['candidates'] if c['text']==fixture()['candidates'][0]['text'])
            value=add_checks(dict(decision='pass',confidence=90,reason='Offline fixture',findings=[],relevant_ids=[cid]), data, settings['model'].startswith('z-ai/'))
        return value,{'model':settings['model'],'attempts':[{'usage':{'cost':.001},'provider':'offline','seconds':.1}]}


class WorkflowTests(unittest.TestCase):
    def test_parallel_workers_do_not_exceed_acceptance_limit(self):
        folder = self.root/'limited'
        specs = [{**SPEC, 'slot_id':f'limited_{i}'} for i in range(3)]
        with patch('workflow.make_plan',return_value=specs):
            prepare(folder,self.source,3,42,pilot=True)
        client = Client()
        out = generate_run(folder,client,limit=1)
        self.assertEqual(out['accepted'],1)
        self.assertEqual(out['statuses']['pending'],2)
        self.assertEqual(client.calls,3)

    def test_parallel_acceptance_rechecks_duplicates(self):
        folder = self.root/'parallel'
        specs = [{**SPEC, 'slot_id':f'parallel_{i}'} for i in range(3)]
        cfg = load_config(); cfg['max_generation_attempts'] = 1
        barrier = threading.Barrier(3)
        def judged(item, *args):
            barrier.wait(timeout=5)
            return {'status':'accepted', 'reason':'fixture', 'family':item, 'events':[], 'repairs':0}
        with patch('workflow.load_config',return_value=cfg), patch('workflow.make_plan',return_value=specs):
            prepare(folder,self.source,3,42,pilot=True)
            with patch('workflow.generate_family',side_effect=lambda *args:(fixture(),{})), patch('workflow.evaluate',side_effect=judged):
                out = generate_run(folder,Client(),limit=3)
        self.assertEqual(out['accepted'],1)
        self.assertEqual(out['statuses']['exhausted'],2)

    def test_parallel_call_budget_and_job_provenance(self):
        db = Store(self.root/'budget.sqlite3')
        budget = {'remaining':2, 'lock':threading.Lock()}
        class RawClient:
            def call(self, settings, prompt):
                return {'ok':True}, {'attempts':[]}
        def call(i):
            client = CachedClient(db,RawClient(),2,job=str(i),budget=budget)
            try:
                client.call({'model':'fixture'},str(i)); return True
            except TransportError:
                return False
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                self.assertEqual(sum(pool.map(call,range(3))),2)
            starts = db.execute("SELECT job,value FROM events WHERE kind='request_started'")
            ends = db.execute("SELECT job,value FROM events WHERE kind='request_finished'")
            self.assertEqual({(j,json.loads(v)['key']) for j,v in starts},
                             {(j,json.loads(v)['key']) for j,v in ends})
            self.assertEqual(budget['remaining'],0)
        finally:
            db.close()

    def test_morph_context_must_be_shared(self):
        item = fixture()
        item['candidates'][0]['text'] = 'İşlem kayda alındı. ' + item['candidates'][0]['text']
        errors = Guard({'texts': [], 'forbidden_lemmas': []}).check(item, SPEC)
        self.assertIn('quality:morph_1:context_changed', errors)
        self.assertIn('quality:morph_2:context_changed', errors)

    def test_morph_hard_cannot_replace_an_unrelated_object(self):
        item = fixture()
        item['candidates'][1]['critical_sentence'] = 'Suna dün pazarda taze çörek otu topladı.'
        item['candidates'][1]['text'] = item['candidates'][1]['critical_sentence']
        errors = Guard({'texts': [], 'forbidden_lemmas': []}).check(item, SPEC)
        self.assertIn('quality:morph_1:non_target_content_drift', errors)

    def test_planned_sentence_counts_are_enforced(self):
        item = fixture(); item['context_sentences'] = ['Nötr bağlam.']
        errors = Guard({'texts': [], 'forbidden_lemmas': []}).check(item, SPEC)
        self.assertIn('quality:shared_context_sentence_count', errors)
        item['context_sentences'] = []
        item['query'] = 'İlk cümle. İkinci cümle.'
        errors = Guard({'texts': [], 'forbidden_lemmas': []}).check(item, SPEC)
        self.assertIn('quality:query_sentence_count', errors)

    def test_strict_pair_cannot_change_non_target_content(self):
        item = fixture();candidate = item['candidates'][1]
        candidate['text'] = candidate['critical_sentence'] = 'Suna dün gölette hiç yüzdü.'
        self.assertIn('quality:morph_1:strict_non_target_edit',
                      Guard({'texts': [], 'forbidden_lemmas': []}).check(item, SPEC))
        loose = {**SPEC, 'family_mode': 'controlled_diverse'}
        self.assertNotIn('quality:morph_1:strict_non_target_edit',
                         Guard({'texts': [], 'forbidden_lemmas': []}).check(item, loose))

    def test_positive_cannot_contain_query_sentence(self):
        protected = {'texts': [], 'forbidden_lemmas': []}
        item = fixture()
        item['candidates'][0]['text'] = 'Bugün hava serin. ' + item['query']
        self.assertIn('quality:query_sentence_copied_into_positive', Guard(protected).check(item, SPEC))
    def test_chain_partition_and_quotas(self):
        from production import HERE
        chains = {f['key'] for f in read_json(HERE/'catalog.json')['features'] if f.get('objective') == 'composition'}
        protected = {'forbidden_features': sorted(chains), 'forbidden_templates': [],
                     'forbidden_domain_register': [], 'chain_protection_version': 2,
                     'chain_forbidden_lemmas': {key: ['korunankök'] for key in chains}}
        plan = make_plan(protected, 1000, 42)
        self.assertEqual(sum(s['generalization_policy'] == 'root_chain_holdout' for s in plan), 300)
        self.assertTrue(chains <= {s['target_feature'] for s in plan})
        protected.pop('chain_forbidden_lemmas')
        with self.assertRaises(ValueError):
            make_plan(protected, 100, 42)

    def test_shared_chain_root_is_blocked(self):
        protected = {'texts': [], 'forbidden_lemmas': [],
                     'chain_forbidden_lemmas': {'NEG.AOR': ['yüz']}}
        self.assertNotIn('leakage:heldout_root_chain_pair', Guard(protected).check(fixture(), SPEC))
        chain_spec = {**SPEC, 'target_feature': 'NEG.AOR'}
        item = fixture(); item['target_feature'] = 'NEG.AOR'
        self.assertIn('leakage:heldout_root_chain_pair', Guard(protected).check(item, chain_spec))
        for candidate in item['candidates']:
            candidate['critical_lemma'] = 'başkakök'
        self.assertNotIn('leakage:heldout_root_chain_pair', Guard(protected).check(item, chain_spec))

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'source.jsonl';self.folder=self.root/'run'
        rows=[dict(family_id=f'protected_{i}',query=f'Arşivdeki belge numarası {i}.',
                   candidates=[{'text':f'Korunan belge {i} bölüm {j} kaydı.'} for j in range(11)],
                   generalization_bucket='standard',domain='finance',register='everyday',
                   target_feature='NEG',template_id='event_report',critical_lemma='arşiv') for i in range(600)]
        self.source.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows))
        with patch('workflow.make_plan',return_value=[SPEC]):
            prepare(self.folder,self.source,1,42)

    def tearDown(self):
        self.tmp.cleanup()

    def approved(self):
        approve(self.folder,read_json(self.folder/'protected.json')['source_sha256'],'offline fixture reviewer')

    def test_final_train_has_no_human_gate(self):
        client=Client()
        self.assertEqual(generate_run(self.folder,client)['accepted'],1)
        self.assertEqual(client.calls,3)

    def test_git_jsonl_shard_round_trip(self):
        client = Client()
        self.assertEqual(generate_run(self.folder, client)['accepted'], 1)
        shard_dir = self.root / 'shards'
        shard = shard_dir / 'codex_001_001.jsonl'
        export_shard(self.folder, shard, 'codex', 1, 1)
        status = validate_shards(self.folder, shard_dir)
        self.assertEqual(status['covered'], 1)
        self.assertEqual(sync_shards(self.folder, shard_dir)['imported'], 0)

    def test_resume_and_export(self):
        self.approved();client=Client()
        self.assertEqual(generate_run(self.folder,client)['accepted'],1)
        self.assertEqual(client.calls,3)
        self.assertEqual(generate_run(self.folder,client)['accepted'],1)
        self.assertEqual(client.calls,3)
        db=Store(self.folder/'state.sqlite3')
        try:
            path=export(self.folder,db)
            row=json.loads(Path(path).read_text())
            self.assertEqual(len(row['training']['negatives']),3)
            self.assertEqual(report(self.folder,db)['known_cost_usd'],.003)
        finally:db.close()

    def test_transport_resume_reuses_generator_and_other_judge(self):
        self.approved();client=Client(fail_once=True)
        self.assertEqual(generate_run(self.folder,client)['accepted'],0)
        self.assertEqual(generate_run(self.folder,client)['accepted'],1)
        self.assertEqual(client.calls,4)

    def test_budget_not_data_rejection(self):
        self.approved();client=Client()
        out=generate_run(self.folder,client,max_calls=1)
        self.assertEqual(out['statuses'],{'pending':1})
        self.assertEqual(client.calls,1)
        self.assertEqual(generate_run(self.folder,client)['accepted'],1)

    def test_source_change_blocks_resume(self):
        self.approved();self.source.write_text(self.source.read_text()+'\n')
        with self.assertRaises(ValueError):verify(self.folder)

    def test_plan_change_blocks_resume(self):
        (self.folder/'plan.json').write_text('[]')
        with self.assertRaises(ValueError):verify(self.folder)

    def test_guard_critical_and_duplicates(self):
        p=read_json(self.folder/'protected.json');guard=Guard(p)
        self.assertEqual(guard.check(fixture(),SPEC),[])
        guard.add(fixture())
        self.assertIn('leakage:protected_or_accepted_text_overlap',guard.check(fixture(),SPEC))
        p['forbidden_lemmas']=['yüz']
        self.assertIn('leakage:heldout_lemma',Guard(p).check(fixture(),SPEC))

    def test_similarity_and_short_exact(self):
        idx=SimilarityIndex(['Rana eski depodaki bütün evrakları dikkatle kutuya yerleştirdi.','Kayıt tamam.'])
        self.assertTrue(idx.overlaps('Rana eski depodaki bütün evrakları özenle kutuya yerleştirdi.'))
        self.assertTrue(idx.overlaps('Kayıt tamam!'))
        self.assertFalse(idx.overlaps('Gölette yüzme eğitimi yarın başlıyor.'))

    def test_holdout_allows_incidental_context_not_target(self):
        p=read_json(self.folder/'protected.json');p['forbidden_lemmas']=['havuz']
        # All fixture texts contain havuz, but their target words are gir/yüz.
        self.assertEqual(Guard(p).check(fixture(),SPEC),[])
        p['forbidden_lemmas']=['yüz']
        self.assertIn('leakage:heldout_lemma',Guard(p).check(fixture(),SPEC))

    def test_pilot_has_no_final_train_eligibility(self):
        folder=self.root/'pilot'
        with patch('workflow.make_plan',return_value=[SPEC]):
            prepare(folder,self.source,1,42,pilot=True)
        self.assertFalse(read_json(folder/'manifest.json')['reviewed'])
        self.assertEqual(generate_run(folder,Client())['accepted'],1)
        db=Store(folder/'state.sqlite3')
        try:
            row=json.loads(Path(export(folder,db)).read_text())
            self.assertEqual(row['purpose'],'pilot_only')
            self.assertFalse(row['eligible_for_final_train'])
        finally:db.close()

    def test_balanced_plan_and_holdout(self):
        p=read_json(self.folder/'protected.json');p['forbidden_features']=['NEG'];p['forbidden_templates']=['formal_record']
        plan=make_plan(p,1000)
        self.assertEqual(plan,make_plan(p,1000))
        self.assertEqual(sum(s['family_mode']=='strict_minimal' for s in plan),400)
        self.assertEqual(sum(s['query_sentence_count']==1 for s in plan),750)
        self.assertTrue(all(s['target_feature']!='NEG' and s['template']['id']!='formal_record' for s in plan))

    def test_lock_prevents_second_writer(self):
        with run_lock(self.folder):
            with self.assertRaises(RuntimeError):
                with run_lock(self.folder):pass


if __name__=='__main__':unittest.main()
