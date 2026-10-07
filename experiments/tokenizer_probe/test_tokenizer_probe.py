import importlib.util
import json
import unittest
from pathlib import Path

from experiments.tokenizer_probe import marked_text as mt
from experiments.tokenizer_probe.data import (ARTIFACTS, SEALED_PATH, SEALED_SHA256, file_sha256, has_mojibake,
                                              load_sealed, repair_text, training_row, training_rows)
from experiments.tokenizer_probe.split import build_split, coverage_rows, validate_split
from test.taxonomy import MORPH_HARD_SUBTYPES, SEMANTIC_HARD_SUBTYPES

ITEMS, REPAIRS = load_sealed()
HAS_TOKENIZERS = all(importlib.util.find_spec(m) for m in ("tokenizers", "transformers"))


class DataTest(unittest.TestCase):
    def test_sealed_file_is_pinned_and_untouched(self):
        self.assertEqual(file_sha256(SEALED_PATH), SEALED_SHA256)
        self.assertEqual(len(ITEMS), 600)

    def test_mojibake_is_repaired_in_memory_only(self):
        self.assertEqual(repair_text("sayÄ±m"), "sayım")
        self.assertEqual(repair_text("Geçen hafta"), "Geçen hafta")
        self.assertEqual(len(REPAIRS["families"]), 3)
        self.assertFalse(any(has_mojibake(t) for it in ITEMS for t in [it["query"]] + [c["text"] for c in it["candidates"]]))
        raw = json.loads(SEALED_PATH.read_text(encoding="utf-8"))["items"]
        self.assertTrue(any(has_mojibake(c["text"]) for it in raw for c in it["candidates"]))

    def test_training_rows_use_two_morph_and_one_semantic_negative(self):
        item = ITEMS[0]
        row = training_row(item)
        by_text = {c["text"].strip(): c for c in item["candidates"]}
        self.assertEqual(by_text[row["positive"]]["role"], "positive")
        self.assertIn(by_text[row["morph_1"]]["subtype"], MORPH_HARD_SUBTYPES)
        self.assertIn(by_text[row["morph_2"]]["subtype"], MORPH_HARD_SUBTYPES)
        self.assertIn(by_text[row["semantic_1"]]["subtype"], SEMANTIC_HARD_SUBTYPES)
        self.assertNotEqual(row["morph_1"], row["morph_2"])
        self.assertEqual(row, training_row(item))
        self.assertEqual(len(training_rows(ITEMS)), 600)


class SplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_split(ITEMS, data_repairs=REPAIRS)

    def test_invariants(self):
        validate_split(self.manifest, ITEMS)
        for fold in self.manifest["folds"]:
            self.assertEqual((len(fold["train"]), len(fold["val"]), len(fold["test"])), (360, 120, 120))

    @unittest.skipUnless((ARTIFACTS / "split_manifest.json").exists(), "artifacts missing")
    def test_committed_manifest_matches_rebuild(self):
        committed = json.loads((ARTIFACTS / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(committed["fold_of"], self.manifest["fold_of"])

    def test_every_phenomenon_is_tested_out_of_fold(self):
        rows = coverage_rows(self.manifest, ITEMS)
        self.assertEqual(len(rows), 76)
        self.assertEqual(sum(r["n_total"] for r in rows), 600)
        self.assertTrue(all(r["train_min"] >= 1 and r["val_min"] >= 0 for r in rows))


class MarkTest(unittest.TestCase):
    def test_marks_only_eligible_words_and_round_trips(self):
        segments = {"beklerken": ["bek", "ler", "ken"], "da": ["d", "a"], "Ankara": ["Anka", "ra"]}
        text = "Ankara'da sıra beklerken, 12 kişi."
        marked = mt.mark_text(text, segments)
        self.assertEqual(marked, f"Anka{mt.MARKER}ra'da sıra bek{mt.MARKER}ler{mt.MARKER}ken, 12 kişi.")
        self.assertEqual(marked.replace(mt.MARKER, ""), text)

    def test_random_and_shifted_controls_keep_counts_and_letters(self):
        word, pieces = "doldurmuştum", ["doldur", "muş", "tum"]
        rand = mt.random_segments(word, 2, seed=1)
        self.assertEqual(("".join(rand), len(rand)), (word, 3))
        self.assertEqual(rand, mt.random_segments(word, 2, seed=1))
        shifted = mt.shifted_segments(word, pieces, seed=1)
        self.assertEqual(("".join(shifted), len(shifted)), (word, 3))
        self.assertNotEqual(shifted, pieces)
        self.assertTrue(all(shifted))

    def test_arms_and_item_marking(self):
        caches = {"tt": {"beklerken": ["bek", "ler", "ken"]}, "mph": {"beklerken": ["bekle", "rken"]}}
        self.assertIsNone(mt.arm_cache("base", caches, []))
        self.assertEqual(mt.arm_cache("mph", caches, []), caches["mph"])
        for arm in ("shift_tt", "rand_mph"):
            cache = mt.arm_cache(arm, caches, ["beklerken"])
            self.assertEqual("".join(cache["beklerken"]), "beklerken")
        marked = mt.mark_items(ITEMS[:5], caches["tt"])
        self.assertEqual([i["family_id"] for i in marked], [i["family_id"] for i in ITEMS[:5]])
        self.assertEqual([c["id"] for c in marked[0]["candidates"]], [c["id"] for c in ITEMS[0]["candidates"]])
        self.assertEqual(mt.item_texts(marked)[0].replace(mt.MARKER, ""), ITEMS[0]["query"])


@unittest.skipUnless((ARTIFACTS / "seg_tt.json").exists() and (ARTIFACTS / "seg_mph.json").exists(), "segmentation caches missing")
class SegmentationCacheTest(unittest.TestCase):
    def test_segments_reproduce_surface_words(self):
        for name, max_fallback in (("tt", 0.10), ("mph", 0.0)):
            payload = json.loads((ARTIFACTS / f"seg_{name}.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["source_sha256"], SEALED_SHA256)
            for word, pieces in payload["segments"].items():
                self.assertEqual("".join(pieces), word)
                self.assertTrue(len(pieces) >= 2 and all(pieces))
            self.assertLessEqual(payload["stats"]["fallback_rate"], max_fallback)


@unittest.skipUnless(HAS_TOKENIZERS, "tokenizers/transformers not installed (use .venv-tok)")
class TokenizerPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        try:
            cls.original = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", local_files_only=True)
            cls.patched = mt.load_patched_tokenizer()
        except OSError:
            raise unittest.SkipTest("mE5-large tokenizer not in the local HF cache")

    def test_unmarked_text_tokenizes_exactly_as_before(self):
        texts = [("query: " + it["query"]) for it in ITEMS] + [("passage: " + c["text"]) for it in ITEMS for c in it["candidates"]]
        self.assertEqual(self.patched(texts)["input_ids"], self.original(texts)["input_ids"])

    def test_marker_never_reaches_the_ids_and_forces_boundaries(self):
        word, pieces = "beklerken", ["bek", "ler", "ken"]
        tokens = self.patched.tokenize(mt.MARKER.join(pieces))
        self.assertFalse(any(mt.MARKER in t for t in tokens))
        flat = [t.replace("▁", "") for t in tokens]
        self.assertEqual("".join(flat), word)
        cuts, total = set(), 0
        for t in flat[:-1]:
            total += len(t)
            cuts.add(total)
        self.assertTrue({3, 6} <= cuts)
        self.assertTrue(tokens[0].startswith("▁") and not any(t.startswith("▁") for t in tokens[1:]))

    def test_every_cached_boundary_is_a_token_boundary(self):
        if not (ARTIFACTS / "seg_mph.json").exists():
            self.skipTest("cache missing")
        segments = json.loads((ARTIFACTS / "seg_mph.json").read_text(encoding="utf-8"))["segments"]
        for word, pieces in list(segments.items())[:500]:
            tokens = [t.replace("▁", "") for t in self.patched.tokenize(mt.MARKER.join(pieces))]
            cuts, total = set(), 0
            for t in tokens[:-1]:
                total += len(t)
                cuts.add(total)
            want, total = set(), 0
            for p in pieces[:-1]:
                total += len(p)
                want.add(total)
            self.assertTrue(want <= cuts, (word, pieces, tokens))


class ReportTest(unittest.TestCase):
    def test_verdict_rule(self):
        from experiments.tokenizer_probe.report import cluster_bootstrap, verdict
        import numpy as np
        self.assertEqual(verdict(-0.05, -0.08, -0.02, 0.01, 0, 5), "zararlı")
        self.assertTrue(verdict(0.05, 0.02, 0.08, 0.01, 5, 5).startswith("etkili"))
        self.assertEqual(verdict(0.0, -0.02, 0.02, 0.9, 2, 5), "pratikte fark yok")
        self.assertTrue(verdict(0.02, -0.03, 0.07, 0.3, 3, 5).startswith("belirsiz"))
        low, high = cluster_bootstrap(np.array([0.1] * 20), ["a"] * 10 + ["b"] * 10, n_boot=200)
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.1)


if __name__ == "__main__":
    unittest.main()
