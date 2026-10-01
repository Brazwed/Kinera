"""Gera vocabulário e exemplos autoregressivos a partir das anotações."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from language.dataset import build_autoregressive_examples, load_jsonl, write_jsonl
from language.vocabulary import CharacterVocabulary, SyllableVocabulary


def build_character_dataset(
    annotated_path: str | Path,
    train_groups_path: str | Path,
    validation_groups_path: str | Path,
    vocabulary_path: str | Path,
    train_output: str | Path,
    validation_output: str | Path,
    report_path: str | Path | None = None,
    *,
    context_size: int = 8,
    syllable_vocabulary_path: str | Path | None = None,
) -> dict[str, Any]:
    records = load_jsonl(annotated_path)
    train_groups = load_jsonl(train_groups_path)
    validation_groups = load_jsonl(validation_groups_path)
    train_keys = {group["word_key"] for group in train_groups}
    validation_keys = {group["word_key"] for group in validation_groups}
    if train_keys & validation_keys:
        raise ValueError("treino e validação compartilham palavras")

    train_records = [record for record in records if record.get("word_key") in train_keys]
    validation_records = [
        record for record in records if record.get("word_key") in validation_keys
    ]
    vocabulary = CharacterVocabulary.from_records(train_records)
    syllable_vocabulary = None
    if syllable_vocabulary_path is not None:
        syllable_vocabulary = SyllableVocabulary.from_dict(
            json.loads(Path(syllable_vocabulary_path).read_text(encoding="utf-8"))
        )
    Path(vocabulary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(vocabulary_path).write_text(
        json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    train_examples = build_autoregressive_examples(
        train_records,
        vocabulary,
        context_size=context_size,
        syllable_vocabulary=syllable_vocabulary,
    )
    validation_examples = build_autoregressive_examples(
        validation_records,
        vocabulary,
        context_size=context_size,
        syllable_vocabulary=syllable_vocabulary,
    )
    write_jsonl(train_output, train_examples)
    write_jsonl(validation_output, validation_examples)

    unknown_validation_targets = sum(
        example["target_char_id"] == vocabulary.unk_id
        for example in validation_examples
    )
    report = {
        "annotated_input": str(annotated_path),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_words": len(train_keys),
        "validation_words": len(validation_keys),
        "vocabulary_size": len(vocabulary),
        "context_size": context_size,
        "syllable_vocabulary": (
            None if syllable_vocabulary_path is None else str(syllable_vocabulary_path)
        ),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "unknown_validation_targets": unknown_validation_targets,
        "outputs": {
            "vocabulary": str(vocabulary_path),
            "train": str(train_output),
            "validation": str(validation_output),
        },
    }
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gera dataset autoregressivo de caracteres.")
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--train-groups", type=Path, required=True)
    parser.add_argument("--validation-groups", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--syllable-vocabulary", type=Path)
    args = parser.parse_args()
    report = build_character_dataset(
        args.annotated,
        args.train_groups,
        args.validation_groups,
        args.vocabulary,
        args.train_output,
        args.validation_output,
        args.report,
        context_size=args.context_size,
        syllable_vocabulary_path=args.syllable_vocabulary,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
