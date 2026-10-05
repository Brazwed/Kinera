"""Run controlled A/B/C/D experiments (default: A/C/D)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinera.experiments.comparison import run_controlled_experiments
from kinera.experiments.config import ExecutionConfig
from kinera.experiments.variants import DEFAULT_VARIANTS, SUPPORTED_VARIANTS
from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--syllable-vocabulary", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=SUPPORTED_VARIANTS,
        default=list(DEFAULT_VARIANTS),
        help="variantes a executar (padrão: A C D)",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help="backend de execução (padrão: cpu)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        execution_config = ExecutionConfig(
            workers=args.workers,
            backend=args.backend,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    vocabulary = CharacterVocabulary.from_dict(
        json.loads(args.vocabulary.read_text(encoding="utf-8"))
    )
    syllable_vocabulary = SyllableVocabulary.from_dict(
        json.loads(args.syllable_vocabulary.read_text(encoding="utf-8"))
    )
    train_examples = load_jsonl(args.train, max_records=args.max_examples)
    validation_examples = load_jsonl(args.validation, max_records=args.max_examples)
    report = run_controlled_experiments(
        train_examples,
        validation_examples,
        vocabulary,
        syllable_vocab_size=len(syllable_vocabulary),
        syllable_unknown_id=syllable_vocabulary.unk_id,
        seeds=args.seeds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        variants=tuple(args.variants),
        execution_config=execution_config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
