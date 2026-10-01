"""Gera vocabulário e exemplos autoregressivos a partir das anotações."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from language.dataset import build_autoregressive_examples, iter_jsonl
from language.vocabulary import CharacterVocabulary, SPECIAL_TOKENS, SyllableVocabulary


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
    train_keys = {group["word_key"] for group in iter_jsonl(train_groups_path)}
    validation_keys = {group["word_key"] for group in iter_jsonl(validation_groups_path)}
    if train_keys & validation_keys:
        raise ValueError("treino e validação compartilham palavras")

    character_tokens: set[str] = set()
    train_records = 0
    validation_records = 0
    for record in iter_jsonl(annotated_path):
        word_key = record.get("word_key")
        if word_key in train_keys:
            train_records += 1
            word = record.get("word_normalized", record.get("word_raw", ""))
            if isinstance(word, str):
                character_tokens.update(word)
        elif word_key in validation_keys:
            validation_records += 1

    vocabulary = CharacterVocabulary([*SPECIAL_TOKENS, *sorted(character_tokens)])
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

    train_example_count = 0
    validation_example_count = 0
    unknown_validation_targets = 0
    unknown_train_syllable_targets = None
    unknown_validation_syllable_targets = None
    if syllable_vocabulary is not None:
        unknown_train_syllable_targets = 0
        unknown_validation_syllable_targets = 0

    train_destination = Path(train_output)
    validation_destination = Path(validation_output)
    train_destination.parent.mkdir(parents=True, exist_ok=True)
    validation_destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        train_destination.open("w", encoding="utf-8", newline="\n") as train_handle,
        validation_destination.open("w", encoding="utf-8", newline="\n") as validation_handle,
    ):
        for record in iter_jsonl(annotated_path):
            word_key = record.get("word_key")
            if word_key not in train_keys and word_key not in validation_keys:
                continue
            examples = build_autoregressive_examples(
                [record],
                vocabulary,
                context_size=context_size,
                syllable_vocabulary=syllable_vocabulary,
            )
            handle = train_handle if word_key in train_keys else validation_handle
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                if word_key in train_keys:
                    train_example_count += 1
                    if syllable_vocabulary is not None:
                        unknown_train_syllable_targets += (
                            example["target_syllable_id"] == syllable_vocabulary.unk_id
                        )
                else:
                    validation_example_count += 1
                    unknown_validation_targets += (
                        example["target_char_id"] == vocabulary.unk_id
                    )
                    if syllable_vocabulary is not None:
                        unknown_validation_syllable_targets += (
                            example["target_syllable_id"] == syllable_vocabulary.unk_id
                        )
    report = {
        "annotated_input": str(annotated_path),
        "train_records": train_records,
        "validation_records": validation_records,
        "train_words": len(train_keys),
        "validation_words": len(validation_keys),
        "vocabulary_size": len(vocabulary),
        "context_size": context_size,
        "syllable_vocabulary": (
            None if syllable_vocabulary_path is None else str(syllable_vocabulary_path)
        ),
        "syllable_targets_persisted": syllable_vocabulary is not None,
        "unknown_train_syllable_targets": unknown_train_syllable_targets,
        "unknown_validation_syllable_targets": unknown_validation_syllable_targets,
        "train_examples": train_example_count,
        "validation_examples": validation_example_count,
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
