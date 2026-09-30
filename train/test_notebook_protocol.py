import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTRUCT_LITERAL = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\\n"
    "Query:"
)


def notebook_source(relative_path: str) -> str:
    notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    return "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )


class NotebookProtocolTest(unittest.TestCase):
    def test_modernbert_query_protocol_matches_frozen_baseline(self):
        paths = [
            "test/notebooks/morph_baseline_eval_600_colab.ipynb",
            "train/notebooks/train_morph_encoder_colab.ipynb",
            "train/notebooks/train_morph_encoder_pilot600_colab.ipynb",
            "train/notebooks/train_morph_encoder_compare5_colab.ipynb",
        ]

        for path in paths:
            with self.subTest(path=path):
                source = notebook_source(path)
                self.assertIn(INSTRUCT_LITERAL, source)
                self.assertRegex(
                    source,
                    r'["\']modernbert-tr["\'].*?["\'](?:q|query_prefix)["\']\s*:\s*INSTRUCT',
                )


if __name__ == "__main__":
    unittest.main()
