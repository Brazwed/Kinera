"""Mede o baseline da Fase 7.1 sem alterar o código de treino."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import resource
import time
from pathlib import Path
from typing import Any, Callable

import kinera.training.evaluator as evaluator_module
import kinera.training.trainer as trainer_module
from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.losses import MultiTaskLoss
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.evaluator import evaluate_multitask_examples
from kinera.training.trainer import train_multitask_examples
from kinera.experiments.comparison import count_trainable_parameters, prepare_variant_examples


STAGES = (
    "data_preparation",
    "preparation_features",
    "forward",
    "loss",
    "backward",
    "optimizer_update",
    "evaluation",
)


def _rss_kb() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(value / 1024)
    return int(value)


def _profile_hotspots(profiler: cProfile.Profile, limit: int = 20) -> list[dict[str, Any]]:
    stats = pstats.Stats(profiler).stats
    ordered = sorted(stats.items(), key=lambda item: item[1][3], reverse=True)
    hotspots: list[dict[str, Any]] = []
    for (filename, line, function), values in ordered[:limit]:
        primitive_calls, total_calls, total_time, cumulative_time, _ = values
        hotspots.append({
            "function": f"{Path(filename).name}:{line}({function})",
            "primitive_calls": primitive_calls,
            "total_calls": total_calls,
            "total_time_seconds": total_time,
            "cumulative_time_seconds": cumulative_time,
        })
    return hotspots


def _timed_method(
    original: Callable[..., Any],
    stage_times: dict[str, int],
    phase: Callable[[], str],
    stage: str,
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            if phase() == "training":
                stage_times[stage] += time.perf_counter_ns() - started

    return wrapped


def profile_workload(
    vocabulary_path: Path,
    syllable_vocabulary_path: Path,
    train_path: Path,
    validation_path: Path,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    max_examples: int,
    context_size: int,
    workers: int,
) -> dict[str, Any]:
    if workers != 1:
        raise ValueError("o baseline da Fase 7.1 exige workers=1")
    stage_times = {stage: 0 for stage in STAGES}
    state = {"phase": "setup"}
    total_started = time.perf_counter_ns()
    profiler = cProfile.Profile()

    profiler.enable()
    try:
        vocabulary = CharacterVocabulary.from_dict(
            json.loads(vocabulary_path.read_text(encoding="utf-8"))
        )
        syllables = SyllableVocabulary.from_dict(
            json.loads(syllable_vocabulary_path.read_text(encoding="utf-8"))
        )
        train = load_jsonl(str(train_path), max_records=max_examples)
        validation = load_jsonl(str(validation_path), max_records=max_examples)

        started = time.perf_counter_ns()
        prepared_train = prepare_variant_examples(
            train, "D", vocabulary, syllable_unknown_id=syllables.unk_id
        )
        prepared_validation = prepare_variant_examples(
            validation, "D", vocabulary, syllable_unknown_id=syllables.unk_id
        )
        stage_times["data_preparation"] += time.perf_counter_ns() - started

        model = AutoregressiveModel(len(vocabulary), len(syllables), seed=seed)
        parameter_count = count_trainable_parameters(model)
        loss_function = MultiTaskLoss(
            char_weight=1.0,
            syllable_weight=0.7,
            boundary_weight=0.3,
        )

        original_example_to_inputs = trainer_module.example_to_inputs
        original_evaluator_example_to_inputs = evaluator_module.example_to_inputs
        original_methods = {
            "forward": model.forward_heads,
            "loss_forward": loss_function.forward,
            "loss_backward": loss_function.backward,
            "heads_backward": model.heads.backward,
            "representation_backward": model.representation.backward,
            "heads_step": model.heads.step,
            "representation_step": model.representation.step,
        }

        def timed_inputs(*args: Any, **kwargs: Any) -> Any:
            started_inputs = time.perf_counter_ns()
            try:
                return original_example_to_inputs(*args, **kwargs)
            finally:
                if state["phase"] == "training":
                    stage_times["preparation_features"] += time.perf_counter_ns() - started_inputs

        trainer_module.example_to_inputs = timed_inputs  # type: ignore[assignment]
        evaluator_module.example_to_inputs = timed_inputs  # type: ignore[assignment]
        model.forward_heads = _timed_method(original_methods["forward"], stage_times, lambda: state["phase"], "forward")  # type: ignore[method-assign]
        loss_function.forward = _timed_method(original_methods["loss_forward"], stage_times, lambda: state["phase"], "loss")  # type: ignore[method-assign]
        loss_function.backward = _timed_method(original_methods["loss_backward"], stage_times, lambda: state["phase"], "backward")  # type: ignore[method-assign]
        model.heads.backward = _timed_method(original_methods["heads_backward"], stage_times, lambda: state["phase"], "backward")  # type: ignore[method-assign]
        model.representation.backward = _timed_method(original_methods["representation_backward"], stage_times, lambda: state["phase"], "backward")  # type: ignore[method-assign]
        model.heads.step = _timed_method(original_methods["heads_step"], stage_times, lambda: state["phase"], "optimizer_update")  # type: ignore[method-assign]
        model.representation.step = _timed_method(original_methods["representation_step"], stage_times, lambda: state["phase"], "optimizer_update")  # type: ignore[method-assign]

        state["phase"] = "evaluation"
        evaluation_initial_started = time.perf_counter_ns()
        initial_validation = evaluate_multitask_examples(
            model, prepared_validation, vocabulary,
            syllable_unknown_id=syllables.unk_id, loss_function=loss_function,
        )
        stage_times["evaluation"] += time.perf_counter_ns() - evaluation_initial_started

        state["phase"] = "training"
        training = train_multitask_examples(
            model, prepared_train, vocabulary,
            epochs=epochs,
            learning_rate=learning_rate,
            syllable_unknown_id=syllables.unk_id,
            loss_function=loss_function,
        )

        state["phase"] = "evaluation"
        evaluation_final_started = time.perf_counter_ns()
        final_validation = evaluate_multitask_examples(
            model, prepared_validation, vocabulary,
            syllable_unknown_id=syllables.unk_id, loss_function=loss_function,
        )
        stage_times["evaluation"] += time.perf_counter_ns() - evaluation_final_started
    finally:
        if "original_example_to_inputs" in locals():
            trainer_module.example_to_inputs = original_example_to_inputs  # type: ignore[assignment]
            evaluator_module.example_to_inputs = original_evaluator_example_to_inputs  # type: ignore[assignment]
        profiler.disable()

    total_ns = time.perf_counter_ns() - total_started
    measured = {
        stage: {
            "nanoseconds": value,
            "seconds": value / 1_000_000_000,
            "percent_total": (100.0 * value / total_ns) if total_ns else 0.0,
        }
        for stage, value in stage_times.items()
    }
    measured_ns = sum(stage_times.values())
    measured["unattributed_overhead"] = {
        "nanoseconds": max(0, total_ns - measured_ns),
        "seconds": max(0, total_ns - measured_ns) / 1_000_000_000,
        "percent_total": max(0.0, 100.0 * (total_ns - measured_ns) / total_ns),
    }
    return {
        "phase": "7.1_performance_baseline",
        "workload": {
            "variant": "D",
            "seed": seed,
            "workers": workers,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "max_examples": max_examples,
            "context_size": context_size,
        },
        "model": {
            "char_vocab_size": len(vocabulary),
            "syllable_vocab_size": len(syllables),
            "parameter_count": parameter_count,
        },
        "metrics": {
            "training": training,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
        },
        "timing": {
            "total_nanoseconds": total_ns,
            "total_seconds": total_ns / 1_000_000_000,
            "stages": measured,
        },
        "peak_rss_kb": _rss_kb(),
        "cprofile_hotspots": _profile_hotspots(profiler),
        "notes": [
            "Medição apenas; nenhuma otimização foi aplicada.",
            "workers=1 é a referência determinística.",
            "A preparação inclui carregamento e materialização da variante D; preparation_features mede a adaptação por exemplo durante o treino.",
            "Evaluation agrega as avaliações inicial e final.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--syllable-vocabulary", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = profile_workload(
        args.vocabulary, args.syllable_vocabulary, args.train, args.validation,
        seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
        max_examples=args.max_examples, context_size=args.context_size,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
