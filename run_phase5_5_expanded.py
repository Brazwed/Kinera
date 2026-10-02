"""Execução ampliada e retomável da análise por classe ortográfica."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import traceback
from pathlib import Path

from language.dataset import load_jsonl
from language.vocabulary import CharacterVocabulary, SyllableVocabulary
from training.experiment_types import ExperimentConfig
from training.orthographic_analysis import run_orthographic_class_analysis
from training.runner_common import aggregate_run_dicts, write_json_checkpoint


def run_seed(payload: dict) -> dict:
    started = time.perf_counter()
    try:
        vocabulary = CharacterVocabulary.from_dict(json.loads(Path(payload['vocabulary']).read_text(encoding='utf-8')))
        syllables = SyllableVocabulary.from_dict(json.loads(Path(payload['syllable_vocabulary']).read_text(encoding='utf-8')))
        report = run_orthographic_class_analysis(
            load_jsonl(payload['train'], max_records=payload['max_examples']),
            load_jsonl(payload['validation'], max_records=payload['max_examples']),
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
    config = ExperimentConfig(
        phase='5.5_orthographic_class_analysis',
        variants=('A', 'B', 'C', 'D'),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_examples=args.max_examples,
        workers=args.workers,
    ).to_dict()
    if args.resume:
        report = json.loads(args.checkpoint.read_text(encoding='utf-8'))
        if any(report.get(key) != value for key, value in config.items() if key != 'seeds'):
            raise SystemExit('checkpoint incompatível')
        if not set(report.get('seeds', [])) <= set(args.seeds):
            raise SystemExit('seeds incompatíveis')
        results = report.get('runs', [])
    else:
        results = []
    completed = {item['seed'] for item in results if item.get('status') == 'completed'}
    pending = [seed for seed in args.seeds if seed not in completed]
    for seed in pending:
        results.append({'seed': seed, 'status': 'pending', 'worker_pid': None})
    report = {**config, 'workers': args.workers, 'runs': results, 'status': 'running'}
    write_json_checkpoint(args.checkpoint, report)
    payloads = [{
        'vocabulary': str(args.vocabulary),
        'syllable_vocabulary': str(args.syllable_vocabulary),
        'train': str(args.train),
        'validation': str(args.validation),
        'seed': seed,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'max_examples': args.max_examples,
    } for seed in pending]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for payload in payloads:
            entry = next(item for item in results if item['seed'] == payload['seed'])
            entry['status'] = 'running'
            futures[pool.submit(run_seed, payload)] = payload['seed']
        write_json_checkpoint(args.checkpoint, report)
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
            except Exception:
                result = {'seed': seed, 'status': 'failed', 'traceback': traceback.format_exc()}
            entry = next(item for item in results if item['seed'] == seed)
            entry.clear(); entry.update(result)
            report = {**config, 'workers': args.workers, 'runs': results, 'status': 'running'}
            write_json_checkpoint(args.checkpoint, report)
    report.update(aggregate_run_dicts(results))
    report['status'] = 'failed' if report['failed_runs'] else 'completed'
    write_json_checkpoint(args.checkpoint, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
