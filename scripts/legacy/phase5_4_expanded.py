"""Execução ampliada e retomável da Fase 5.4."""

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
from kinera.experiments.multi_seed import run_seed_jobs
from kinera.experiments.partial import run_partial_comparison


def run_seed(payload: dict) -> dict:
    started = time.perf_counter()
    try:
        vocabulary = CharacterVocabulary.from_dict(json.loads(Path(payload['vocabulary']).read_text(encoding='utf-8')))
        syllables = SyllableVocabulary.from_dict(json.loads(Path(payload['syllable_vocabulary']).read_text(encoding='utf-8')))
        report = run_partial_comparison(
            load_jsonl(payload['train'], max_records=payload['max_examples']),
            load_jsonl(payload['validation'], max_records=payload['max_examples']),
            vocabulary,
            syllables,
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


def aggregate(results: list[dict]) -> dict:
    completed = [item for item in results if item.get('status') == 'completed']
    variants = ('A', 'B', 'C', 'D', 'E', 'F')
    output = {'completed_seeds': len(completed)}
    for variant in variants:
        metrics = {}
        for name in ('char_loss', 'char_accuracy', 'total_loss'):
            values = [item['run']['metrics'][variant]['validation'][name] for item in completed]
            if not values:
                continue
            mean = sum(values) / len(values)
            metrics[name] = {
                'mean': mean,
                'std': (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
            }
        output[variant] = metrics
    return output


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
        phase='5.4_expanded_partial_syllable',
        variants=('A', 'B', 'C', 'D', 'E', 'F'),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
    )
    execution_config = ExecutionConfig(workers=args.workers)

    def job_factory(seed: int) -> dict:
        return {
            'vocabulary': str(args.vocabulary),
            'syllable_vocabulary': str(args.syllable_vocabulary),
            'train': str(args.train),
            'validation': str(args.validation),
            'seed': seed,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'max_examples': args.max_examples,
        }

    report = run_seed_jobs(
        seeds=args.seeds,
        experiment_config=experiment_config.to_dict(),
        execution_config=execution_config.to_dict(),
        checkpoint_path=args.checkpoint,
        worker=run_seed,
        job_factory=job_factory,
        aggregate=aggregate,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
