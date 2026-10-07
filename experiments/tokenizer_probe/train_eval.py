"""LoRA fine-tuning and evaluation of the tokenizer-probe arms on mE5-large.

Mirrors the C5 notebook protocol (``train/notebooks/train_morph_encoder_compare5_colab.ipynb``,
cells 4 and 8): same LoRA, loss, optimiser, prefixes and best-epoch restore. Only the tokenization
of the text differs between arms. Needs a GPU for the real run; ``--smoke`` works anywhere.

    python -m experiments.tokenizer_probe.train_eval --stage zero_shot --arms base tt mph
    python -m experiments.tokenizer_probe.train_eval --stage folds --arms base tt mph --folds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import time
import traceback
from pathlib import Path
from typing import Any

from .data import ARTIFACTS, PURPOSE, RUNS, SEALED_SHA256, load_items, training_rows
from .marked_text import ARMS, arm_cache, item_texts, load_caches, mark_items, patch_tokenizer, unique_words

MODEL = "intfloat/multilingual-e5-large"
QUERY_PREFIX, DOC_PREFIX = "query: ", "passage: "
LORA_TARGETS = ["query", "key", "value", "dense"]
# C5 values; epochs chosen so the step budget (about 140) stays close to C5's 112.
PARAMS = dict(epochs=6, batch_size=32, cache_mini_batch=16, learning_rate=1.5e-5, warmup_ratio=.1,
              weight_decay=.01, lora_r=16, lora_alpha=32, lora_dropout=.05, max_seq_length=256, scale=20.0)
NEGATIVE_SEED = 42  # negatives are data, not a training seed: identical across arms and seeds
SMOKE_SIZES = (48, 24, 24)
DEVICE: str | None = None  # "cpu" forces CPU (MPS has no dropout SDPA, so local smoke tests need it)


def environment() -> dict[str, Any]:
    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("transformers", "sentence_transformers", "peft"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # pragma: no cover - diagnostics only
            info[name] = None
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return info


def load_model(model_name: str, params: dict[str, Any]):
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available() and DEVICE != "cpu":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = SentenceTransformer(model_name, device=DEVICE, model_kwargs={"torch_dtype": dtype})
    model.max_seq_length = params["max_seq_length"]
    patch_tokenizer(model.tokenizer)
    return model


def marked_families(arm: str, items: list[dict[str, Any]], caches, seed: int = 42) -> dict[str, dict[str, Any]]:
    segments = arm_cache(arm, caches, unique_words(item_texts(items)), seed)
    return {item["family_id"]: item for item in mark_items(items, segments)}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def zero_shot(arms, items, caches, model_name, params, out_dir: Path, force: bool) -> None:
    from test.evaluation import score_encoder

    model = None
    for arm in arms:
        path = out_dir / "zero_shot" / f"{arm}.json"
        if path.exists() and not force:
            print(f"[skip] {path}")
            continue
        model = model or load_model(model_name, params)
        marked = list(marked_families(arm, items, caches).values())
        started = time.time()
        result = score_encoder(model, marked, query_prefix=QUERY_PREFIX, document_prefix=DOC_PREFIX, batch_size=32)
        _write(path, {"purpose": PURPOSE, "arm": arm, "stage": "zero_shot", "model": model_name,
                      "source_sha256": SEALED_SHA256, "summary": result["summary"], "per_query": result["per_query"],
                      "seconds": round(time.time() - started, 1), "env": environment()})
        print(f"[zero_shot] {arm}: recall@1={result['summary']['recall@1']:.3f} mrr@10={result['summary']['mrr@10']:.3f}")


def train_rows_dataset(rows: list[dict[str, str]]):
    from datasets import Dataset

    samples = []
    for r in rows:
        negatives = {"negative_1": DOC_PREFIX + r["morph_1"], "negative_2": DOC_PREFIX + r["morph_2"],
                     "negative_3": DOC_PREFIX + r["semantic_1"]}
        samples.append({"anchor": QUERY_PREFIX + r["query"], "positive": DOC_PREFIX + r["positive"], **negatives})
        samples.append({"anchor": QUERY_PREFIX + r["positive"], "positive": DOC_PREFIX + r["query"], **negatives})
    return Dataset.from_list(samples)


def run_fold(arm, fold, seed, items, caches, manifest, model_name, params, out_dir: Path, force: bool, smoke: bool):
    import numpy as np
    import pandas as pd
    import torch
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.evaluation import SentenceEvaluator
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    try:
        from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    except ImportError:  # older sentence-transformers
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from test.evaluation import score_encoder

    path = out_dir / arm / f"fold{fold}_seed{seed}.json"
    if path.exists() and not force:
        print(f"[skip] {path}")
        return
    split = manifest["folds"][fold]
    train_ids, val_ids, test_ids = split["train"], split["val"], split["test"]
    if smoke:
        train_ids, val_ids, test_ids = (ids[:n] for ids, n in zip((train_ids, val_ids, test_ids), SMOKE_SIZES))
    marked = marked_families(arm, items, caches)
    train_items, val_items, test_items = ([marked[i] for i in ids] for ids in (train_ids, val_ids, test_ids))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.time()
    print(f"\n===== arm={arm} fold={fold} seed={seed} train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} =====")

    model = load_model(model_name, params)
    first = model._first_module()
    available = {name.rsplit(".", 1)[-1] for name, _ in first.auto_model.named_modules()}
    targets = [t for t in LORA_TARGETS if t in available]
    if not targets:
        raise RuntimeError(f"LoRA hedefi bulunamadı: {sorted(available)[:40]}")

    def score(subset):
        return score_encoder(model, subset, query_prefix=QUERY_PREFIX, document_prefix=DOC_PREFIX, batch_size=32)

    val_zero = score(val_items)["summary"]
    first.auto_model = get_peft_model(first.auto_model, LoraConfig(
        r=params["lora_r"], lora_alpha=params["lora_alpha"], lora_dropout=params["lora_dropout"], bias="none",
        target_modules=targets, task_type="FEATURE_EXTRACTION"))
    # A fresh LoRA adapter must not change anything; otherwise the comparison is not interpretable.
    init_val = score(val_items)["summary"]
    peft_init_delta = {k: init_val[k] - val_zero[k] for k in ("recall@1", "recall@3", "mrr@10", "ndcg@10")}
    peft_init_ok = all(abs(v) < 1e-6 for v in peft_init_delta.values())
    print("LoRA başlangıç kontrolü:", "OK" if peft_init_ok else f"UYARI {peft_init_delta}")

    class ValEvaluator(SentenceEvaluator):
        def __init__(self):
            super().__init__()
            self.name = "val"
            self.primary_metric = "val_mrr_at_10"
            self.greater_is_better = True

        def __call__(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):
            summary = score_encoder(model, val_items, query_prefix=QUERY_PREFIX, document_prefix=DOC_PREFIX,
                                    batch_size=32)["summary"]
            return {"val_recall_at_1": summary["recall@1"], "val_mrr_at_10": summary["mrr@10"]}

    class RestoreBestLoRA(TrainerCallback):
        def __init__(self, metric):
            self.metric, self.best_score, self.best_state, self.best_epoch = metric, float("-inf"), None, None

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            score_value = (metrics or {}).get(f"eval_{self.metric}", (metrics or {}).get(self.metric))
            if score_value is not None and score_value > self.best_score:
                self.best_score, self.best_epoch = float(score_value), state.epoch
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
            return control

    cuda = torch.cuda.is_available() and DEVICE != "cpu"
    bf16 = cuda and torch.cuda.is_bf16_supported()
    evaluator, best = ValEvaluator(), RestoreBestLoRA("val_mrr_at_10")
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "_tmp"), num_train_epochs=1 if smoke else params["epochs"],
        per_device_train_batch_size=params["batch_size"], learning_rate=params["learning_rate"],
        warmup_ratio=params["warmup_ratio"], weight_decay=params["weight_decay"], fp16=cuda and not bf16, bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES, logging_strategy="steps", logging_steps=5, logging_first_step=True,
        eval_strategy="epoch", save_strategy="no", load_best_model_at_end=False, eval_on_start=True,
        seed=seed, data_seed=seed, report_to="none", use_cpu=DEVICE == "cpu")
    loss = CachedMultipleNegativesRankingLoss(model, scale=params["scale"], mini_batch_size=params["cache_mini_batch"])
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_rows_dataset(training_rows(train_items, NEGATIVE_SEED)),
        loss=loss, evaluator=evaluator, callbacks=[best])
    trainer.train()
    if best.best_state:
        trainer.model.load_state_dict(best.best_state, strict=False)
    model = trainer.model
    val_best = score(val_items)["summary"]
    test = score(test_items)
    _write(path, {
        "purpose": PURPOSE, "arm": arm, "stage": "lora", "fold": fold, "seed": seed, "smoke": smoke,
        "model": model_name, "params": params, "lora_targets": targets, "source_sha256": SEALED_SHA256,
        "n": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "best_epoch": best.best_epoch, "best_val_mrr_at_10": best.best_score,
        "peft_init_ok": peft_init_ok, "peft_init_delta": peft_init_delta,
        "val_zero": val_zero, "val_best": val_best, "test_summary": test["summary"], "per_query": test["per_query"],
        "training_log": pd.DataFrame(trainer.state.log_history).to_dict("records"),
        "seconds": round(time.time() - started, 1), "env": environment()})
    print(f"[done] {arm} fold{fold} seed{seed}: best_epoch={best.best_epoch} val_mrr={best.best_score:.3f} "
          f"test recall@1={test['summary']['recall@1']:.3f} mrr@10={test['summary']['mrr@10']:.3f} "
          f"({time.time() - started:.0f}s)")
    del trainer, loss, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("zero_shot", "folds", "all"), default="all")
    parser.add_argument("--arms", nargs="+", default=["base", "tt", "mph"], choices=ARMS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--epochs", type=int, default=PARAMS["epochs"])
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--out", type=Path, default=RUNS)
    parser.add_argument("--smoke", action="store_true", help="fold 0 only, tiny subsets, 1 epoch")
    parser.add_argument("--cpu", action="store_true", help="force CPU (local smoke tests)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    global DEVICE
    DEVICE = "cpu" if args.cpu else None
    params = {**PARAMS, "epochs": args.epochs}
    items = load_items()
    caches = load_caches(args.artifacts)
    manifest = json.loads((args.artifacts / "split_manifest.json").read_text(encoding="utf-8"))
    if args.smoke:
        args.out = args.out / "_smoke"
        args.folds = args.folds[:1]
    if args.stage in ("zero_shot", "all"):
        zero_shot(args.arms, items, caches, args.model, params, args.out, args.force)
    failures = []
    if args.stage in ("folds", "all"):
        for seed in args.seeds:
            for fold in args.folds:
                for arm in args.arms:
                    try:
                        run_fold(arm, fold, seed, items, caches, manifest, args.model, params, args.out,
                                 args.force, args.smoke)
                    except Exception:
                        traceback.print_exc()
                        failures.append((arm, fold, seed))
    if failures:
        raise SystemExit(f"Başarısız koşular: {failures}")


if __name__ == "__main__":
    main()
