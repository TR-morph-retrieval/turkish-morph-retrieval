"""Offline tests; no credentials, network, test data or test imports."""
import json
from copy import deepcopy
import unittest
from unittest.mock import patch
from io import BytesIO
from production import (load_config, policy, validate_family, evaluate, TransportError,
                        OpenRouter, judge_prompt, FACT_KEYS, materialize, checked_verdict,
                        canonicalize_candidate_annotations)


def report(decision='pass', confidence=88, findings=None):
    return dict(decision=decision, confidence=confidence, reason='Somut kontrol sonucu', findings=findings or [])


FAMILY = dict(query='Bora tutarı geri göndermedi.', target_feature='NEG',
              event_frame={k: 'unspecified' for k in FACT_KEYS}, context_sentences=[], critical_position=0,
              target_description='Eylemin gerçekleşmemesi', candidates=[
                  dict(slot='positive', text='Bora parayı iade etmedi.'),
                  dict(slot='morph_1', text='Bora parayı iade etti.'),
                  dict(slot='morph_2', text='Bora parayı iade etmeyecek.'),
                  dict(slot='semantic_1', text='Ece parayı iade etmedi.')])
for candidate in FAMILY['candidates']:
    candidate['critical_sentence'] = candidate['text']
    candidate['critical_lemma'] = 'iade'
    candidate['critical_pos'] = 'VERB'
    if candidate['slot'].startswith('morph_'):
        candidate['morph_change'] = {'feature':'NEG', 'from':'negative', 'to':'affirmative'}


def add_checks(verdict, data, morphology=False):
    keys = ('target_valid', 'target_feature_match', 'natural', 'content_preserved') if morphology else FACT_KEYS
    if not morphology:
        verdict['positive_fact_coverage'] = {k: True for k in FACT_KEYS}
        verdict['query_claims'] = {k: 'fixture' for k in FACT_KEYS}
        verdict['positive_claims'] = {k: 'fixture' for k in FACT_KEYS}
    verdict['candidate_checks'] = [
        {'candidate_id': c['candidate_id'],
         'checks': {k: (True if morphology or k != 'event' else c['candidate_id'] in verdict.get('relevant_ids', [])) for k in keys},
         **({'observed_feature': 'NEG karşıtlığı'} if morphology else {}),
         **({} if morphology else {'natural': True}),
         'evidence': 'Synthetic fixture comparison'} for c in data['candidates']]
    return verdict


class FakeClient:
    def __init__(self, mode='pass'):
        self.mode, self.patches = mode, 0

    def call(self, settings, prompt):
        if prompt.startswith('Türkçe retrieval train family içindeki yalnız belirtilen aday slotlarını düzelt.'):
            self.patches += 1
            patches = [{'slot': 'morph_1', 'critical_sentence': 'Bora parayı dün iade etti.'}]
            if self.mode == 'unauthorized_patch':
                patches.append({'slot': 'positive', 'critical_sentence': 'Yetkisiz değişiklik.'})
            return {'patches': patches}, {}
        if self.mode == 'transport':
            raise TransportError('offline failure')
        data = json.loads(prompt.split('Veri:\n')[1])
        positive = next(c['candidate_id'] for c in data['candidates'] if c['text'] == FAMILY['candidates'][0]['text'])
        wrong = next(c['candidate_id'] for c in data['candidates'] if c['text'] in {'Bora parayı iade etti.', 'Bora parayı dün iade etti.'})
        v = report()
        v['relevant_ids'] = [positive]
        if self.mode == 'uncertain':
            v['confidence'] = 79
        if self.mode == 'blind_mismatch':
            v['relevant_ids'] = [positive, wrong]
        if self.mode in {'repair', 'unauthorized_patch', 'always_fail'} and (not self.patches or self.mode == 'always_fail'):
            v = report('fail', 90, [{'candidate_id': wrong, 'reason': 'Hatalı aday'}])
            v['relevant_ids'] = [positive]
        return add_checks(v, data, settings['model'].startswith('z-ai/')), {'requested_model': settings['model']}


class ProductionTests(unittest.TestCase):
    def test_missing_and_contradictory_checks_cannot_pass(self):
        ids = {'c0':'positive'}
        v = report(); v['relevant_ids'] = ['c0']
        self.assertEqual(checked_verdict(v, 'semantic', ids), {})

    def test_location_drift_cannot_be_a_semantic_pass(self):
        ids = {'c0':'positive', 'c1':'morph_1'}
        v = report(); v['relevant_ids'] = ['c0']
        v['candidate_checks'] = []
        for cid in ids:
            checks = {k: True for k in FACT_KEYS}
            if cid == 'c0':
                checks['place'] = False  # query=sığınak, candidate=zirve
            else:
                checks['event'] = False
            v['candidate_checks'].append({'candidate_id':cid, 'checks':checks,
                                          'natural': True,
                                          'evidence':'Sığınak yerine zirve yazılmış.'})
        self.assertEqual(checked_verdict(v, 'semantic', ids), {})
        add_checks(v, {'candidates':[{'candidate_id':'c0'}]})
        v['candidate_checks'][0]['checks']['place'] = False
        self.assertEqual(checked_verdict(v, 'semantic', ids), {})
        v['candidate_checks'][0]['checks']['place'] = None
        self.assertEqual(checked_verdict(v, 'semantic', ids), {})

    def test_morphology_false_check_overrides_pass(self):
        v = report(); add_checks(v, {'candidates':[{'candidate_id':'c0'}]}, True)
        v['candidate_checks'][0]['checks']['content_preserved'] = False
        self.assertEqual(checked_verdict(v, 'morphology', {'c0':'morph_1'})['decision'], 'fail')

    def test_wrong_realized_target_feature_overrides_pass(self):
        v = report(); add_checks(v, {'candidates':[{'candidate_id':'c0'}]}, True)
        v['candidate_checks'][0]['checks']['target_feature_match'] = False
        v['candidate_checks'][0]['observed_feature'] = 'Hedef PL yerine olumsuzluk değişmiş.'
        checked = checked_verdict(v, 'morphology', {'c0':'morph_1'})
        self.assertEqual(checked['decision'], 'fail')
        self.assertEqual(checked['findings'][0]['candidate_id'], 'c0')

    def test_context_materialized_once_and_repairable(self):
        f = deepcopy(FAMILY); f['context_sentences'] = ['Nötr bağlam.']; f['critical_position'] = 1
        out = materialize(f)
        self.assertTrue(all(c['text'] == 'Nötr bağlam. ' + c['critical_sentence'] for c in out['candidates']))
        self.assertEqual(materialize(out), out)

    def test_context_ambiguity_requires_the_same_surface_word(self):
        f = deepcopy(FAMILY)
        f['target_feature'] = 'MORPH.CONTEXT_AMBIG'
        f['candidates'][0]['critical_word'] = 'yazar'
        f['candidates'][1]['critical_word'] = 'yazar'
        f['candidates'][2]['critical_word'] = 'yazdı'
        self.assertIn('morph_2:context_ambiguity_surface_not_preserved', validate_family(f))
    def test_role_visibility_is_scoped_to_morphology(self):
        semantic = json.loads(judge_prompt(FAMILY, 'semantic', FAMILY['candidates']).split('Veri:\n')[1])
        morphology = json.loads(judge_prompt(FAMILY, 'morphology', FAMILY['candidates']).split('Veri:\n')[1])
        self.assertTrue(all('slot' not in c for c in semantic['candidates']))
        self.assertEqual(morphology['candidates'][0]['slot'], 'positive')

    def test_threshold(self):
        self.assertEqual(policy({'semantic':report(confidence=80), 'morphology':report(confidence=80)}, {'c0'})['action'], 'accept')
        self.assertEqual(policy({'semantic':report(confidence=78), 'morphology':report()}, {'c0'}, 80, 85)['action'], 'human_review')
        self.assertEqual(policy({'semantic':report('abstain'), 'morphology':report()}, {'c0'}, 80, 85)['action'], 'retry')

    def test_widespread_failure_regenerates_instead_of_patching_every_slot(self):
        findings = [{'candidate_id': f'c{i}', 'reason': 'Yaygın hata'} for i in range(3)]
        bad = report('fail', 90, findings)
        decision = policy({'semantic': bad, 'morphology': report()}, {'c0','c1','c2'})
        self.assertEqual(decision['action'], 'regenerate')

    def test_pass_fail_disagreement_is_accepted_with_review_flag(self):
        class DisagreeingClient(FakeClient):
            def call(self, settings, prompt):
                value, provenance = super().call(settings, prompt)
                if settings['model'].startswith('z-ai/') and 'decision' in value:
                    cid = value['candidate_checks'][1]['candidate_id']
                    value['candidate_checks'][1]['checks']['natural'] = False
                    value['decision'] = 'fail'
                    value['findings'] = [{'candidate_id': cid, 'reason': 'Doğallık kuşkusu'}]
                return value, provenance
        out = evaluate(FAMILY, DisagreeingClient(), load_config(), lambda x: [])
        self.assertEqual(out['status'], 'accepted')
        self.assertEqual(out['train_decision'], 'human_review')
        self.assertEqual(out['review_reason'], 'judge_decision_disagreement')

    def test_naturalness_disagreement_is_review_flagged(self):
        ids = {'c0': 'positive'}
        semantic = report(); semantic['candidate_checks'] = [{'candidate_id': 'c0', 'natural': False}]
        morphology = report(); morphology['candidate_checks'] = [{'candidate_id': 'c0', 'checks': {'natural': True}}]
        self.assertEqual(policy({'semantic': semantic, 'morphology': morphology}, ids, 80, 85),
                         {'action': 'human_review', 'reason': 'judge_naturalness_disagreement'})

    def test_invalid_reports(self):
        for bad in [{}, report(confidence=True), report(confidence=float('nan')), report('fail'),
                    report(findings=[{'candidate_id':'c0', 'reason':'oops'}])]:
            self.assertEqual(policy({'semantic':bad, 'morphology':report()}, {'c0'})['action'], 'retry')

    def test_structure(self):
        self.assertEqual(validate_family(FAMILY), [])
        x=deepcopy(FAMILY); x['query']='GeÃ§en hafta.'
        self.assertIn('text:encoding_or_control_character', validate_family(x))
        x=deepcopy(FAMILY); x['candidates'][0]['slot']=[]
        self.assertTrue(validate_family(x))

    def test_accept_and_guard(self):
        self.assertEqual(evaluate(FAMILY, FakeClient(), load_config(), lambda x:[])['status'], 'accepted')
        self.assertEqual(evaluate(FAMILY, FakeClient(), load_config(), lambda x:['leakage'])['status'], 'rejected')

    def test_repair_only_flagged(self):
        out=evaluate(FAMILY, FakeClient('repair'), load_config(), lambda x:[])
        self.assertEqual(out['status'], 'accepted'); self.assertEqual(out['repairs'], 1)
        self.assertEqual(out['family']['query'], FAMILY['query'])
        self.assertEqual(out['family']['candidates'][0], FAMILY['candidates'][0])
        self.assertEqual(FAMILY['candidates'][1]['text'], 'Bora parayı iade etti.')

    def test_local_pair_heuristic_warns_without_spending_a_repair(self):
        calls = 0
        def guard(_):
            nonlocal calls
            calls += 1
            return ['quality:morph_1:strict_non_target_edit'] if calls == 1 else []
        out = evaluate(FAMILY, FakeClient(), load_config(), guard)
        self.assertEqual(out['status'], 'accepted')
        self.assertEqual(out['repairs'], 0)
        self.assertEqual(out['train_decision'], 'human_review')
        self.assertIn('quality:morph_1:strict_non_target_edit', out['local_quality_warnings'])
        self.assertEqual(out['family']['candidates'][0], FAMILY['candidates'][0])

    def test_unflagged_patch_is_ignored(self):
        out = evaluate(FAMILY, FakeClient('unauthorized_patch'), load_config(), lambda x:[])
        self.assertEqual(out['status'], 'accepted')
        repair = next(e for e in out['events'] if e.get('stage') == 'repair')
        self.assertEqual(repair['ignored_unrequested_slots'], ['positive'])
        self.assertEqual(out['family']['candidates'][0], FAMILY['candidates'][0])

    def test_bounded_repairs(self):
        out=evaluate(FAMILY, FakeClient('always_fail'), load_config(), lambda x:[])
        self.assertEqual(out['status'], 'rejected'); self.assertEqual(out['repairs'], 1)

    def test_adequately_confident_pass_is_not_retried(self):
        self.assertEqual(evaluate(FAMILY, FakeClient('uncertain'), load_config(), lambda x:[])['status'], 'accepted')

    def test_low_confidence_double_pass_enters_train_with_review_metadata(self):
        class LowConfidenceClient(FakeClient):
            def call(self, settings, prompt):
                value, provenance = super().call(settings, prompt)
                if 'decision' in value:
                    value['confidence'] = 65
                return value, provenance
        out = evaluate(FAMILY, LowConfidenceClient(), load_config(), lambda x:[])
        self.assertEqual(out['status'], 'accepted')
        self.assertEqual(out['train_decision'], 'human_review')
        self.assertEqual(out['review_reason'], 'low_confidence_pass')
        self.assertEqual(out['judge_confidence'], {'semantic': 65, 'morphology': 65})

    def test_moderately_confident_double_pass_is_review_flagged(self):
        semantic = report(); morphology = report()
        semantic['confidence'] = morphology['confidence'] = 72
        self.assertEqual(policy({'semantic': semantic, 'morphology': morphology},
                                {'c0': 'positive'}, 80, 85)['action'], 'human_review')

    def test_soft_local_pair_warning_enters_train_with_review_metadata(self):
        out = evaluate(FAMILY, FakeClient(), load_config(),
                       lambda _: ['quality:morph_1:strict_non_target_edit',
                                  'morph_2:positive_critical_lemma_mismatch'])
        self.assertEqual(out['status'], 'accepted')
        self.assertEqual(out['train_decision'], 'human_review')
        self.assertEqual(out['review_reason'], 'local_quality_warning')
        self.assertEqual(out['local_quality_warnings'], [
            'quality:morph_1:strict_non_target_edit',
            'morph_2:positive_critical_lemma_mismatch'])

    def test_candidate_annotation_is_canonicalized_from_its_text(self):
        f = deepcopy(FAMILY)
        f['candidates'][0]['critical_word'] = 'etmedi'
        f['candidates'][0]['critical_sentence'] = 'Bora parayı iade etmedi !'
        canonicalize_candidate_annotations(f)
        self.assertEqual(f['candidates'][0]['critical_sentence'], 'Bora parayı iade etmedi.')

    def test_blind_relevance(self):
        self.assertNotEqual(evaluate(FAMILY, FakeClient('blind_mismatch'), load_config(), lambda x:[])['status'], 'accepted')

    def test_transport(self):
        self.assertEqual(evaluate(FAMILY, FakeClient('transport'), load_config(), lambda x:[])['status'], 'deferred_transport')

    def test_length_and_price_payload(self):
        def response(finish, content):
            return BytesIO(json.dumps({'choices':[{'finish_reason':finish,'message':{'content':content}}],
                                      'model':'google/gemini-3.8-flash', 'provider':'mock',
                                      'usage':{'cost':0.01}}).encode())
        with patch('production.urlopen', side_effect=[response('length','{}'), response('stop','{"ok":true}')]) as call:
            value, provenance=OpenRouter('fake').call(load_config()['generator'], 'offline')
        self.assertTrue(value['ok'])
        self.assertEqual(len(provenance['attempts']), 2)
        first=json.loads(call.call_args_list[0].args[0].data)
        second=json.loads(call.call_args_list[1].args[0].data)
        self.assertEqual(first['provider']['sort'], 'price')
        self.assertNotIn('require_parameters', first['provider'])
        self.assertEqual(second['max_tokens'], first['max_tokens']*2)

    def test_length_exhaustion_is_transport(self):
        def response(*args, **kwargs):
            return BytesIO(b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}')
        with patch('production.urlopen', side_effect=response) as call:
            with self.assertRaises(TransportError):
                OpenRouter('fake').call(load_config()['generator'], 'offline')
        self.assertEqual(call.call_count, 3)


if __name__ == '__main__':
    unittest.main()
