"""Execução ampliada e retomável da análise por classe ortográfica."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import traceback
from pathlib import Path

from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.experiments.config import ExecutionConfig, ExperimentConfig
from kinera.experiments.fingerprint import (
    build_legacy_experiment_fingerprint,
    build_legacy_scientific_configuration,
    load_legacy_experiment_inputs,
    validate_legacy_checkpoint_identity,
)
from kinera.experiments.multi_seed import run_seed_jobs
from kinera.diagnostics.orthography import run_orthographic_class_analysis


def run_seed(payload: dict) -> dict:
    started = time.perf_counter()
    try:
        train, validation, vocabulary, syllables, _ = load_legacy_experiment_inputs(
            payload,
            phase='5.5_orthographic_class_analysis',
            require_expected_identity=True,
        )
        report = run_orthographic_class_analysis(
            train,
            validation,
            vocabulary,
            syllable_vocab_size=len(syllables),
            syllable_unknown_id=syllables.unk_id,
            seeds=[payload['seed']],
            epochs=payload['epochs'],
            learning_rate=payload['learning_rate'],
            max_examples=payload['max_examples'],
        )
        return {
            'seed': payload['seed'],
            'status': 'completed',
            'worker_pid': os.getpid(),
            'wall_time_seconds': time.perf_counter() - started,
            'run': report['runs'][0],
        }
    except Exception:
        return {
            'seed': payload['seed'],
            'status': 'failed',
            'worker_pid': os.getpid(),
            'wall_time_seconds': time.perf_counter() - started,
            'traceback': traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--syllable-vocabulary', type=Path, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 42, 123])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-examples', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    experiment_config = ExperimentConfig(
        phase='5.5_orthographic_class_analysis',
        variants=('A', 'B', 'C', 'D'),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
    )
    execution_config = ExecutionConfig(workers=args.workers)
    vocabulary = CharacterVocabulary.from_dict(
        json.loads(args.vocabulary.read_text(encoding='utf-8'))
    )
    syllables = SyllableVocabulary.from_dict(
        json.loads(args.syllable_vocabulary.read_text(encoding='utf-8'))
    )
    scientific_configuration = build_legacy_scientific_configuration(
        experiment_config.to_dict(),
        vocabulary=vocabulary,
        syllable_vocabulary=syllables,
    )
    train = load_jsonl(args.train, max_records=args.max_examples)
    validation = load_jsonl(args.validation, max_records=args.max_examples)
    experiment_identity = build_legacy_experiment_fingerprint(
        phase='5.5_orthographic_class_analysis',
        train_examples=train,
        validation_examples=validation,
        vocabulary=vocabulary,
        syllable_vocabulary=syllables,
        scientific_configuration=scientific_configuration,
    )
    identity_payload = {
        'vocabulary': str(args.vocabulary),
        'syllable_vocabulary': str(args.syllable_vocabulary),
        'train': str(args.train),
        'validation': str(args.validation),
        'max_examples': args.max_examples,
        'scientific_configuration': scientific_configuration,
        'experiment_identity': experiment_identity,
    }
    checkpoint_configuration = experiment_config.to_dict()
    checkpoint_configuration['experiment_identity'] = experiment_identity
    if args.resume:
        validate_legacy_checkpoint_identity(
            str(args.checkpoint),
            experiment_identity,
            phase='5.5_orthographic_class_analysis',
        )

    def job_factory(seed: int) -> dict:
        return {
            **identity_payload,
            'seed': seed,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
        }

    report = run_seed_jobs(
        seeds=args.seeds,
        experiment_config=checkpoint_configuration,
        execution_config=execution_config.to_dict(),
        checkpoint_path=args.checkpoint,
        worker=run_seed,
        job_factory=job_factory,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
