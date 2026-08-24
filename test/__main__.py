"""Command-line interface for the test benchmark pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .ranges import range_status, run_range, validate_shared_ranges
from .dataset_memory import DatasetMemory
from .evaluation import load_items
from .exports import finalize
from .judge_report import judge_calibration_report
from .pipeline import (
    default_run_id,
    export_shared_shard,
    generate,
    paths_for,
    read_jsonl,
    sync_shared_shards,
    write_plan,
)
from .review import apply_human_reviews, export_human_review
from .selftest import run as run_selftest
from .validators import artifact_report, train_test_leakage_problems


def _load_flexible(path: str) -> list[dict]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    return load_items(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="varsayılan: test/config.json")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="API çağrısı yapmadan kapsam planı üret")
    plan.add_argument("--run-id", default="planned_test_v39")
    plan.add_argument("--size", type=int, default=None)

    generation = sub.add_parser("generate", help="üret + QC + iki-aşamalı judge; kalmayan slotları refill et")
    generation.add_argument("--run-id", default=None)
    generation.add_argument("--limit", type=int, default=None, help="seçili kuyruktan üretilecek slot sayısı")
    generation.add_argument("--offset", type=int, default=0, help="seçili generator kuyruğundaki başlangıç")
    generation.add_argument("--workers", type=int, default=None)
    generation.add_argument(
        "--generator-id",
        default=None,
        help="planı değiştirmeden yalnız bu generator'a atanmış slotları seç",
    )

    shared_sync = sub.add_parser(
        "shared-sync", help="Git'teki contributor shard'larını accepted state ve SQLite'a aktar"
    )
    shared_sync.add_argument("--run-id", required=True)
    shared_sync.add_argument("--input-dir", default="test/data/final_shards")

    shard_export = sub.add_parser(
        "shard-export", help="tamamlanan generator aralığını paylaşılabilir JSONL shard'a çıkar"
    )
    shard_export.add_argument("--run-id", required=True)
    shard_export.add_argument("--output", required=True)
    shard_export.add_argument("--generator-id", required=True)
    shard_export.add_argument("--offset", type=int, required=True)
    shard_export.add_argument("--limit", type=int, required=True)

    range_show = sub.add_parser(
        "range-show", help="1-based inclusive sıradaki üretim aralığını çakışmadan doğrula"
    )
    range_show.add_argument("--producer", choices=["codex", "claude"], required=True)
    range_show.add_argument("--from", dest="start", type=int, required=True)
    range_show.add_argument("--to", dest="end", type=int, required=True)

    range_run = sub.add_parser(
        "range-run", help="shared sync + sıradaki aralığı üret + manifestli shard export"
    )
    range_run.add_argument("--producer", choices=["codex", "claude"], required=True)
    range_run.add_argument("--from", dest="start", type=int, required=True)
    range_run.add_argument("--to", dest="end", type=int, required=True)
    range_run.add_argument("--workers", type=int, default=None)

    sub.add_parser("ranges-report", help="paylaşılan shard sırası ve kapsamını doğrula")

    freeze = sub.add_parser(
        "finalize",
        help="final human review tamamlanmış 100 dev + 500 sealed testi dondur",
    )
    freeze.add_argument("--run-id", required=True)

    audit = sub.add_parser("audit", help="accepted veya frozen internal veri üzerinde ucuz artefakt denetimi")
    audit.add_argument("--run-id", required=True)

    leakage = sub.add_parser("audit-leakage", help="train ile test arasında exact/fuzzy kopya ara")
    leakage.add_argument("--test", required=True)
    leakage.add_argument("--train", required=True)

    morph = sub.add_parser("morph-audit", help="opsiyonel Stanza lemma/UFeats denetimi")
    morph.add_argument("--input", required=True)
    morph.add_argument("--output", required=True)
    morph.add_argument("--download-model", action="store_true")
    morph.add_argument("--use-gpu", action="store_true")

    memory_report = sub.add_parser(
        "memory-report", help="dataset memory slot durumlarını ve aggregate kapsamı göster"
    )
    memory_report.add_argument("--run-id", required=True)

    memory_ingest = sub.add_parser(
        "memory-ingest", help="train/dev metadata'sını generation memory'ye ekle"
    )
    memory_ingest.add_argument("--run-id", required=True)
    memory_ingest.add_argument("--input", required=True)
    memory_ingest.add_argument("--source", required=True)
    memory_ingest.add_argument("--split", default=None)

    review_export = sub.add_parser(
        "review-export", help="600 accepted family için freeze öncesi kör insan review manifesti üret"
    )
    review_export.add_argument("--run-id", required=True)

    review_apply = sub.add_parser(
        "review-apply", help="final insan kararlarını uygula; red olan slotlar refill için açılır"
    )
    review_apply.add_argument("--run-id", required=True)
    review_apply.add_argument("--input", required=True)

    judge_report = sub.add_parser(
        "judge-report", help="cascade judge, order ve human-calibration metriklerini raporla"
    )
    judge_report.add_argument("--run-id", required=True)

    sub.add_parser("self-test", help="API kullanmadan regression testleri")
    args = parser.parse_args()

    if args.command == "plan":
        result = write_plan(args.run_id, args.config, args.size)
    elif args.command == "generate":
        run_id = args.run_id or default_run_id()
        result = generate(
            run_id, args.config, args.limit, args.workers, args.generator_id, args.offset
        )
        result["run_id"] = run_id
    elif args.command == "finalize":
        result = finalize(args.run_id, args.config)
    elif args.command == "audit":
        run = paths_for(args.run_id)
        result = artifact_report(read_jsonl(run.accepted))
    elif args.command == "audit-leakage":
        cfg = load_config(args.config, runtime=False)
        problems = train_test_leakage_problems(
            _load_flexible(args.test), _load_flexible(args.train), cfg
        )
        result = {"problem_count": len(problems), "problems": problems}
    elif args.command == "morph-audit":
        from .morphology import run_morphology_audit
        result = run_morphology_audit(
            args.input, args.output, args.download_model, args.use_gpu
        )
    elif args.command == "memory-report":
        result = DatasetMemory(paths_for(args.run_id).memory).report()
    elif args.command == "memory-ingest":
        rows = _load_flexible(args.input)
        if args.split:
            rows = [
                {**row, "target_split": row.get("target_split", row.get("split", args.split))}
                for row in rows
            ]
        memory = DatasetMemory(paths_for(args.run_id).memory)
        result = {
            "ingested": memory.ingest_families(rows, args.source),
            "memory": memory.report(),
        }
    elif args.command == "shared-sync":
        result = sync_shared_shards(args.run_id, args.input_dir, args.config)
    elif args.command == "shard-export":
        result = export_shared_shard(
            args.run_id, args.output, args.generator_id, args.offset, args.limit, args.config
        )
    elif args.command == "range-show":
        result = range_status(args.producer, args.start, args.end, config_path=args.config)
    elif args.command == "range-run":
        result = run_range(
            args.producer, args.start, args.end, config_path=args.config, workers=args.workers
        )
    elif args.command == "ranges-report":
        result = validate_shared_ranges(config_path=args.config)
    elif args.command == "review-export":
        result = export_human_review(args.run_id)
    elif args.command == "review-apply":
        result = apply_human_reviews(args.run_id, args.input, args.config)
    elif args.command == "judge-report":
        result = judge_calibration_report(args.run_id)
    else:
        failures = run_selftest()
        if failures:
            for failure in failures:
                print("FAIL:", failure)
            return 1
        result = {"status": "ok", "message": "test pipeline self-test passed"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
