"""Train the autoregressive Kinera model from prepared JSONL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_examples, generate_text
from kinera.training.trainer import train_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--syllable-vocabulary", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    vocabulary = CharacterVocabulary.from_dict(
        json.loads(args.vocabulary.read_text(encoding="utf-8"))
    )
    syllable_vocabulary = None
    if args.syllable_vocabulary is not None:
        syllable_vocabulary = SyllableVocabulary.from_dict(
            json.loads(args.syllable_vocabulary.read_text(encoding="utf-8"))
        )
    examples = load_jsonl(args.train, max_records=args.max_examples)
    model = AutoregressiveModel(
        char_vocab_size=len(vocabulary),
        syllable_vocab_size=1 if syllable_vocabulary is None else len(syllable_vocabulary),
        seed=args.seed,
    )
    report = train_examples(
        model,
        examples,
        vocabulary,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        syllable_unknown_id=(0 if syllable_vocabulary is None else syllable_vocabulary.unk_id),
        max_examples=args.max_examples,
    )
    if args.validation is not None:
        validation_examples = load_jsonl(args.validation, max_records=args.max_examples)
        report["validation"] = evaluate_examples(
            model,
            validation_examples,
            vocabulary,
            syllable_unknown_id=(0 if syllable_vocabulary is None else syllable_vocabulary.unk_id),
            max_examples=args.max_examples,
        )
    report["generated_text"] = generate_text(
        model,
        vocabulary,
        context_size=args.context_size,
        max_tokens=args.max_tokens,
        syllable_unknown_id=(0 if syllable_vocabulary is None else syllable_vocabulary.unk_id),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
