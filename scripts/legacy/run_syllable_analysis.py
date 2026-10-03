"""Executa a análise diagnóstica do ganho silábico da Fase 5.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.diagnostics.syllables import run_syllable_gain_analysis


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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.vocabulary.open(encoding="utf-8") as handle:
        vocabulary = CharacterVocabulary.from_dict(json.load(handle))
    with args.syllable_vocabulary.open(encoding="utf-8") as handle:
        syllable_vocabulary = SyllableVocabulary.from_dict(json.load(handle))
    report = run_syllable_gain_analysis(
        load_jsonl(args.train, max_records=args.max_examples),
        load_jsonl(args.validation, max_records=args.max_examples),
        vocabulary,
        syllable_vocab_size=len(syllable_vocabulary),
        syllable_unknown_id=syllable_vocabulary.unk_id,
        seeds=args.seeds,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        top_examples=args.top_examples,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
