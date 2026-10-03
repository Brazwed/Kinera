"""Execução ampliada e retomável da análise da Fase 5.3."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.experiments.config import ExecutionConfig, ExperimentConfig
from kinera.experiments.multi_seed import run_seed_jobs
from kinera.diagnostics.syllables import run_syllable_gain_analysis


def _run_seed(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        vocabulary = CharacterVocabulary.from_dict(
            json.loads(Path(payload["vocabulary"]).read_text(encoding="utf-8"))
        )
        syllable_vocabulary = SyllableVocabulary.from_dict(
            json.loads(Path(payload["syllable_vocabulary"]).read_text(encoding="utf-8"))
        )
        train = load_jsonl(payload["train"], max_records=payload["max_examples"])
        validation = load_jsonl(payload["validation"], max_records=payload["max_examples"])
        report = run_syllable_gain_analysis(
            train,
            validation,
            vocabulary,
            syllable_vocab_size=len(syllable_vocabulary),
            syllable_unknown_id=syllable_vocabulary.unk_id,
            seeds=[payload["seed"]],
            epochs=payload["epochs"],
            learning_rate=payload["learning_rate"],
            max_examples=payload["max_examples"],
            top_examples=payload["top_examples"],
        )
        run = report["runs"][0]
        run["worker_pid"] = os.getpid()
        run["wall_time_seconds"] = time.perf_counter() - started
        return {"seed": payload["seed"], "status": "completed", "run": run}
    except Exception:
        return {
            "seed": payload["seed"],
            "status": "failed",
            "worker_pid": os.getpid(),
            "wall_time_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == "completed"]
    result: dict[str, Any] = {"completed_seeds": len(completed)}
    for pair in ("A_vs_C", "B_vs_D"):
        values = [run["run"]["pair_deltas"][pair]["overall"] for run in completed]
        if not values:
            continue
        result[pair] = {
            "mean_delta_loss": sum(item["mean_delta_loss"] for item in values) / len(values),
            "accuracy_delta": sum(item["accuracy_delta"] for item in values) / len(values),
            "improved_loss_fraction": sum(item["improved_loss_fraction"] for item in values) / len(values),
            "regressed_loss_fraction": sum(item["regressed_loss_fraction"] for item in values) / len(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--syllable-vocabulary", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--top-examples", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    experiment_config = ExperimentConfig(
        phase="5.3_expanded_analysis",
        variants=("A", "B", "C", "D"),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        top_examples=args.top_examples,
    )
    execution_config = ExecutionConfig(workers=args.workers)

    def job_factory(seed: int) -> dict[str, Any]:
        return {
            "vocabulary": str(args.vocabulary),
            "syllable_vocabulary": str(args.syllable_vocabulary),
            "train": str(args.train),
            "validation": str(args.validation),
            "seed": seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_examples": args.max_examples,
            "top_examples": args.top_examples,
        }

    report = run_seed_jobs(
        seeds=args.seeds,
        experiment_config=experiment_config.to_dict(),
        execution_config=execution_config.to_dict(),
        checkpoint_path=args.checkpoint,
        worker=_run_seed,
        job_factory=job_factory,
        aggregate=_aggregate,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
