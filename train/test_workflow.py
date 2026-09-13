"""Offline end-to-end train tests. Synthetic fixtures are never training data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from production import TransportError
from workflow import (Guard, SimilarityIndex, Store, approve, contract, export, generate_run,
                      make_plan, prepare, read_json, report, run_lock, snapshot, verify)


SPEC = {'slot_id':'train_000001','target_feature':'NEG','target_description':'Eylem gerçekleşmez',
        'feature':{'key':'NEG'},'domain':'daily_life','register':'everyday',
        'template':{'id':'event_report','description':'doğal olay bildirimi'},
        'family_mode':'strict_minimal','query_sentence_count':1,'passage_sentence_count':1}


def fixture():
    item = dict(query='Suna dün havuzda suya girmedi.', query_critical_word='girmedi',
                query_critical_lemma='gir', query_critical_sentence='Suna dün havuzda suya girmedi.',
                target_feature='NEG',target_description='Eylem gerçekleşmez',template_id='event_report',
                domain='daily_life',register='everyday',candidates=[])
    for slot, text, word in [
        ('positive','Suna dün havuzda hiç yüzmedi.','yüzmedi'),
        ('morph_1','Suna dün havuzda hiç yüzdü.','yüzdü'),
        ('morph_2','Suna yarın havuzda hiç yüzmeyecek.','yüzmeyecek'),
        ('semantic_1','Yelda dün gölette hiç yüzmedi.','yüzmedi')]:
        item['candidates'].append(dict(slot=slot,text=text,critical_sentence=text,critical_word=word,critical_lemma='yüz'))
    return item


class Client:
    def __init__(self, fail_once=False):
        self.calls=0;self.fail_once=fail_once

    def call(self, settings, prompt):
        self.calls+=1
        if settings['model'].startswith('google/'):
            value=fixture()
        else:
            if settings['model'].startswith('deepseek/') and self.fail_once:
                self.fail_once=False
                raise TransportError('temporary fixture failure')
            data=json.loads(prompt.split('Veri:\n')[1])
            cid=next(c['candidate_id'] for c in data['candidates'] if c['text']==fixture()['candidates'][0]['text'])
            value=dict(decision='pass',confidence=90,reason='Offline fixture',findings=[],relevant_ids=[cid])
        return value,{'model':settings['model'],'attempts':[{'usage':{'cost':.001},'provider':'offline','seconds':.1}]}


class WorkflowTests(unittest.TestCase):
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

    def test_unapproved_no_calls(self):
        client=Client()
        with self.assertRaises(ValueError):generate_run(self.folder,client)
        self.assertEqual(client.calls,0)

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
