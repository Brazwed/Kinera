"""Executa os experimentos controlados A/B/C/D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from language.dataset import load_jsonl
from language.vocabulary import CharacterVocabulary, SyllableVocabulary
from training.comparison import run_controlled_experiments


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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="arquivo de checkpoint incremental por variante/seed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retoma combinações concluídas a partir do checkpoint",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="quantidade de processos; 1 mantém execução sequencial",
    )
    args = parser.parse_args()

    with args.vocabulary.open(encoding="utf-8") as handle:
        vocabulary = CharacterVocabulary.from_dict(json.load(handle))
    with args.syllable_vocabulary.open(encoding="utf-8") as handle:
        syllable_vocabulary = SyllableVocabulary.from_dict(json.load(handle))
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
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
