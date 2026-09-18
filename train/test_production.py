"""Offline tests; no credentials, network, test data or test imports."""
import json
from copy import deepcopy
import unittest
from unittest.mock import patch
from io import BytesIO
from production import load_config, policy, validate_family, evaluate, TransportError, OpenRouter, judge_prompt


def report(decision='pass', confidence=88, findings=None):
    return dict(decision=decision, confidence=confidence, reason='Somut kontrol sonucu', findings=findings or [])


FAMILY = dict(query='Bora tutarı geri göndermedi.', target_feature='NEG',
              target_description='Eylemin gerçekleşmemesi', candidates=[
                  dict(slot='positive', text='Bora parayı iade etmedi.'),
                  dict(slot='morph_1', text='Bora parayı iade etti.'),
                  dict(slot='morph_2', text='Bora parayı iade etmeyecek.'),
                  dict(slot='semantic_1', text='Ece parayı iade etmedi.')])


class FakeClient:
    def __init__(self, mode='pass'):
        self.mode, self.patches = mode, 0

    def call(self, settings, prompt):
        if settings['model'].startswith('google/'):
            self.patches += 1
            slot = 'positive' if self.mode == 'unauthorized_patch' else 'morph_1'
            return {'patches': [{'slot': slot, 'text': 'Bora parayı dün iade etti.'}]}, {}
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
        return v, {'requested_model': settings['model']}


class ProductionTests(unittest.TestCase):
    def test_role_visibility_is_scoped_to_morphology(self):
        semantic = json.loads(judge_prompt(FAMILY, 'semantic', FAMILY['candidates']).split('Veri:\n')[1])
        morphology = json.loads(judge_prompt(FAMILY, 'morphology', FAMILY['candidates']).split('Veri:\n')[1])
        self.assertTrue(all('slot' not in c for c in semantic['candidates']))
        self.assertEqual(morphology['candidates'][0]['slot'], 'positive')

    def test_threshold(self):
        self.assertEqual(policy({'semantic':report(confidence=80), 'morphology':report(confidence=80)}, {'c0'})['action'], 'accept')
        self.assertEqual(policy({'semantic':report(confidence=79), 'morphology':report()}, {'c0'})['action'], 'accept')

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

    def test_unflagged_patch(self):
        self.assertEqual(evaluate(FAMILY, FakeClient('unauthorized_patch'), load_config(), lambda x:[])['reason'], 'invalid_repair_patch')

    def test_bounded_repairs(self):
        out=evaluate(FAMILY, FakeClient('always_fail'), load_config(), lambda x:[])
        self.assertEqual(out['status'], 'rejected'); self.assertEqual(out['repairs'], 1)

    def test_uncertain(self):
        self.assertEqual(evaluate(FAMILY, FakeClient('uncertain'), load_config(), lambda x:[])['status'], 'accepted')

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
        self.assertTrue(first['provider']['require_parameters'])
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
