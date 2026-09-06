"""Offline regression tests for bounded judge recovery and existing run contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .providers import OpenRouterProvider, ProviderError
from . import pipeline
from .config import load_config
from .planner import build_plan


class TransportTests(unittest.TestCase):
    def test_length_then_success_and_cache(self):
        budgets = []
        def respond(req, **kwargs):
            body = json.loads(req.data)
            budgets.append(body['max_tokens'])
            raw = {'model': 'judge', 'provider': 'route', 'usage': {'cost': 0.01},
                   'choices': [{'finish_reason': 'length' if len(budgets) == 1 else 'stop',
                                'message': {'content': '{"ok": true}'}}]}
            from io import BytesIO
            return BytesIO(json.dumps(raw).encode())
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'TEST_KEY': 'fake'}):
            provider = OpenRouterProvider({'model': 'judge', 'api_key_env': 'TEST_KEY',
                'base_url': 'https://example.invalid', 'max_tokens': 2600}, Path(tmp), {})
            with patch('urllib.request.urlopen', side_effect=respond), patch.object(provider.limiter, 'wait'):
                response = provider.call_json('s', 'p', {}, 'judge')
                self.assertEqual(budgets, [2600, 5200])
                self.assertEqual(response.usage['effective_max_tokens'], 5200)
                self.assertEqual(len(response.usage['transport_attempts']), 2)
                self.assertTrue(provider.call_json('s', 'p', {}, 'judge').cache_hit)

    def test_exhaustion_never_accepts_truncated_json(self):
        from io import BytesIO
        raw = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"ok":true}'}}]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'TEST_KEY': 'fake'}):
            provider = OpenRouterProvider({'model': 'judge', 'api_key_env': 'TEST_KEY',
                'base_url': 'https://example.invalid', 'max_tokens': 2600}, Path(tmp), {})
            with patch('urllib.request.urlopen', side_effect=lambda *a, **k: BytesIO(json.dumps(raw).encode())) as call, patch.object(provider.limiter, 'wait'):
                with self.assertRaises(ProviderError):
                    provider.call_json('s', 'p', {}, 'judge')
                self.assertEqual(call.call_count, 3)
                self.assertFalse(list((Path(tmp) / 'openrouter').glob('*.json')))

    def test_existing_contract_resume_and_unknown_edit_rejected(self):
        cfg = load_config(runtime=False)
        slots = build_plan(cfg)
        contract = json.loads((pipeline.HERE / 'data/final_shards/claude_121_135.jsonl.manifest.json').read_text())['run_contract']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = pipeline.RunPaths(root, root/'plan.json', root/'run_manifest.json',
                root/'accepted.jsonl', root/'rejected.jsonl', root/'failures.jsonl',
                root/'report.json', root/'cache', root/'memory.sqlite3')
            with patch.object(pipeline, 'paths_for', return_value=paths):
                pipeline.initialise_run('final_v39', cfg, slots)
                manifest = json.loads(paths.manifest.read_text())
                for key, value in contract.items():
                    self.assertEqual(manifest[key], value, key)
                pipeline.initialise_run('final_v39', cfg, slots)
                altered = pipeline._pipeline_source_hashes()
                altered['prompts.py'] = 'unapproved'
                with patch.object(pipeline, '_pipeline_source_hashes', return_value=altered):
                    with self.assertRaises(ValueError):
                        pipeline.initialise_run('final_v39', cfg, slots)


if __name__ == '__main__':
    unittest.main()
