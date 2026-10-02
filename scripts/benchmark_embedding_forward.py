"""Paired microbenchmark for Embedding.forward() candidates; no production edits."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from language.dataset import load_jsonl  # noqa: E402
from language.vocabulary import CharacterVocabulary, SyllableVocabulary  # noqa: E402
from nn.embedding import Embedding  # noqa: E402
from training.comparison import prepare_variant_examples  # noqa: E402


def _validate_ids(ids: list[int], vocab_size: int) -> None:
    for position, index in enumerate(ids):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"ids[{position}] deve ser um inteiro")
        if not 0 <= index < vocab_size:
            raise ValueError(f"ids[{position}] fora do vocabulário")


def forward_a(embedding: Embedding, ids: list[int]) -> list[list[float]]:
    """A: current production implementation."""
    return embedding.forward(ids)


def forward_b(embedding: Embedding, ids: list[int]) -> list[list[float]]:
    """B: cache dimensions locally and build independent zero rows by repetition."""
    vocab_size = embedding.vocab_size
    embedding_dim = embedding.embedding_dim
    _validate_ids(ids, vocab_size)

    embedding.last_ids = ids.copy()
    weights = embedding.weights
    embedding.output = [weights[index].copy() for index in ids]
    embedding.grad_weights = [
        [0.0] * embedding_dim for _ in range(vocab_size)
    ]
    embedding._has_backward = False
    return [vector.copy() for vector in embedding.output]


def forward_c_used_rows_only(
    embedding: Embedding,
    ids: list[int],
) -> list[list[float]]:
    """C: reset only rows named in this call; intentionally experimental."""
    vocab_size = embedding.vocab_size
    embedding_dim = embedding.embedding_dim
    _validate_ids(ids, vocab_size)

    embedding.last_ids = ids.copy()
    weights = embedding.weights
    embedding.output = [weights[index].copy() for index in ids]
    for index in dict.fromkeys(ids):
        embedding.grad_weights[index] = [0.0] * embedding_dim
    embedding._has_backward = False
    return [vector.copy() for vector in embedding.output]


VARIANTS: dict[str, Callable[[Embedding, list[int]], list[list[float]]]] = {
    "A": forward_a,
    "B": forward_b,
    "C": forward_c_used_rows_only,
}
BLOCK_ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")


def _actual_d_id_batches(sample_count: int) -> tuple[list[list[int]], list[list[int]], dict[str, object]]:
    processed = ROOT / "data/processed"
    char_vocabulary = CharacterVocabulary.from_dict(
        json.loads((processed / "vocabulary.json").read_text(encoding="utf-8"))
    )
    syllable_vocabulary = SyllableVocabulary.from_dict(
        json.loads((processed / "syllable_vocabulary.json").read_text(encoding="utf-8"))
    )
    raw = load_jsonl(str(processed / "autoregressive_train.jsonl"), max_records=sample_count)
    prepared = prepare_variant_examples(
        raw,
        "D",
        char_vocabulary,
        syllable_unknown_id=syllable_vocabulary.unk_id,
    )
    char_batches = [row["context_ids"] for row in prepared]
    syllable_batches = [row["context_syllable_ids"] for row in prepared]
    context_sizes = {len(ids) for ids in char_batches + syllable_batches}
    if len(context_sizes) != 1:
        raise ValueError(f"context sizes inesperados: {sorted(context_sizes)}")
    stats: dict[str, object] = {
        "variant": "D",
        "source": "first records from data/processed/autoregressive_train.jsonl, prepared as D",
        "sample_contexts": len(prepared),
        "context_size": next(iter(context_sizes)),
        "char_vocab_size": len(char_vocabulary),
        "syllable_vocab_size": len(syllable_vocabulary),
        "embedding_dim": 8,
        "char_unique_ids_per_context": {
            "min": min(len(set(ids)) for ids in char_batches),
            "max": max(len(set(ids)) for ids in char_batches),
            "mean": statistics.mean(len(set(ids)) for ids in char_batches),
        },
        "syllable_unique_ids_per_context": {
            "min": min(len(set(ids)) for ids in syllable_batches),
            "max": max(len(set(ids)) for ids in syllable_batches),
            "mean": statistics.mean(len(set(ids)) for ids in syllable_batches),
        },
    }
    return char_batches, syllable_batches, stats


def _new_embedding(vocab_size: int, embedding_dim: int, seed: int) -> Embedding:
    return Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, seed=seed)


def _forward_snapshot(
    embedding: Embedding,
    returned: list[list[float]],
) -> dict[str, object]:
    return {
        "returned": [row.copy() for row in returned],
        "output": None if embedding.output is None else [row.copy() for row in embedding.output],
        "last_ids": None if embedding.last_ids is None else embedding.last_ids.copy(),
        "grad_weights": [row.copy() for row in embedding.grad_weights],
        "has_backward": embedding._has_backward,
    }


def _copy_contract(embedding: Embedding, ids: list[int], returned: list[list[float]]) -> dict[str, bool]:
    assert embedding.output is not None
    assert embedding.last_ids is not None
    return {
        "last_ids_is_copy": embedding.last_ids is not ids,
        "stored_output_rows_do_not_alias_weights": all(
            out_row is not embedding.weights[index]
            for out_row, index in zip(embedding.output, ids)
        ),
        "returned_rows_do_not_alias_stored_output": all(
            result_row is not stored_row
            for result_row, stored_row in zip(returned, embedding.output)
        ),
        "returned_rows_do_not_alias_weights": all(
            result_row is not embedding.weights[index]
            for result_row, index in zip(returned, ids)
        ),
    }


def _one_call_contract(
    vocab_size: int,
    embedding_dim: int,
    seed: int,
    ids: list[int],
) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, bool]]]:
    reference: dict[str, object] | None = None
    equality: dict[str, dict[str, bool]] = {}
    copy_contract: dict[str, dict[str, bool]] = {}
    for name, fn in VARIANTS.items():
        embedding = _new_embedding(vocab_size, embedding_dim, seed)
        result = fn(embedding, ids)
        state = _forward_snapshot(embedding, result)
        if name == "A":
            reference = state
        assert reference is not None
        equality[name] = {
            key + "_exact_vs_A": state[key] == reference[key]
            for key in ("returned", "output", "last_ids", "grad_weights", "has_backward")
        }
        copy_contract[name] = _copy_contract(embedding, ids, result)
    return equality, copy_contract


def _tracemalloc_probe(
    vocab_size: int,
    embedding_dim: int,
    seed: int,
    ids: list[int],
    fn: Callable[[Embedding, list[int]], list[list[float]]],
) -> dict[str, int]:
    embedding = _new_embedding(vocab_size, embedding_dim, seed)
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    fn(embedding, ids)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "live_bytes_delta": current - before,
        "peak_bytes_above_start": peak - before,
    }


def _stale_gradient_probe(
    vocab_size: int,
    embedding_dim: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    first_ids = list(range(8))
    second_ids = list(range(8, 16))
    learning_rate = 0.05
    results: dict[str, dict[str, object]] = {}
    reference_weights: list[list[float]] | None = None
    for name, fn in VARIANTS.items():
        embedding = _new_embedding(vocab_size, embedding_dim, seed)
        fn(embedding, first_ids)
        embedding.backward([[1.0] * embedding_dim for _ in first_ids])
        embedding.step(learning_rate)
        stale_id = first_ids[0]
        weight_after_first_step = embedding.weights[stale_id].copy()

        fn(embedding, second_ids)
        embedding.backward([[0.0] * embedding_dim for _ in second_ids])
        stale_gradient_before_second_step = embedding.grad_weights[stale_id].copy()
        embedding.step(learning_rate)
        state = [row.copy() for row in embedding.weights]
        if name == "A":
            reference_weights = state
        assert reference_weights is not None
        results[name] = {
            "stale_id": stale_id,
            "stale_gradient_after_next_forward_backward": stale_gradient_before_second_step,
            "stale_weight_delta_on_second_step": [
                after - before
                for after, before in zip(embedding.weights[stale_id], weight_after_first_step)
            ],
            "weights_exact_vs_A_after_two_batches": state == reference_weights,
        }
    return results


def _timer_sample(
    fn: Callable[[Embedding, list[int]], list[list[float]]],
    embedding: Embedding,
    id_batches: list[list[int]],
    cycles: int,
) -> dict[str, int]:
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for _ in range(cycles):
        for ids in id_batches:
            fn(embedding, ids)
    return {
        "wall_ns": time.perf_counter_ns() - wall_started,
        "cpu_ns": time.process_time_ns() - cpu_started,
    }


def _benchmark_shape(
    vocab_size: int,
    embedding_dim: int,
    id_batches: list[list[int]],
    *,
    cycles: int,
    warmups: int,
    seed: int,
) -> dict[str, object]:
    representative_ids = id_batches[0]
    equality, copies = _one_call_contract(vocab_size, embedding_dim, seed, representative_ids)
    allocations = {
        name: _tracemalloc_probe(vocab_size, embedding_dim, seed, representative_ids, fn)
        for name, fn in VARIANTS.items()
    }
    stale_probe = _stale_gradient_probe(vocab_size, embedding_dim, seed)

    warmup_ids = id_batches[: min(16, len(id_batches))]
    warmup_order: list[str] = []
    for _ in range(warmups):
        for name, fn in VARIANTS.items():
            embedding = _new_embedding(vocab_size, embedding_dim, seed)
            for ids in warmup_ids:
                fn(embedding, ids)
            warmup_order.append(name)

    blocks: list[dict[str, object]] = []
    final_states: dict[str, list[dict[str, object]]] = {name: [] for name in VARIANTS}
    for block_index, order in enumerate(BLOCK_ORDERS, start=1):
        embeddings = {
            name: _new_embedding(vocab_size, embedding_dim, seed) for name in VARIANTS
        }
        block: dict[str, object] = {"block": block_index, "order": list(order), "samples": {}}
        samples = block["samples"]
        assert isinstance(samples, dict)
        for name in order:
            emb = embeddings[name]
            sample = _timer_sample(VARIANTS[name], emb, id_batches, cycles)
            samples[name] = sample
            assert emb.output is not None and emb.last_ids is not None
            final_states[name].append({
                "output": [row.copy() for row in emb.output],
                "last_ids": emb.last_ids.copy(),
                "grad_weights": [row.copy() for row in emb.grad_weights],
                "has_backward": emb._has_backward,
            })
        blocks.append(block)

    ratios: dict[str, dict[str, list[float]]] = {
        name: {"wall": [], "cpu": []} for name in ("B", "C")
    }
    for block in blocks:
        samples = block["samples"]
        assert isinstance(samples, dict)
        a = samples["A"]
        assert isinstance(a, dict)
        for name in ("B", "C"):
            value = samples[name]
            assert isinstance(value, dict)
            ratios[name]["wall"].append(a["wall_ns"] / value["wall_ns"])
            ratios[name]["cpu"].append(a["cpu_ns"] / value["cpu_ns"])

    repeated_state_exact = {
        name: [state == final_states["A"][i] for i, state in enumerate(states)]
        for name, states in final_states.items()
    }
    medians = {
        name: {
            clock: statistics.median(
                block["samples"][name][f"{clock}_ns"] / (cycles * len(id_batches))  # type: ignore[index]
                for block in blocks
            )
            for clock in ("wall", "cpu")
        }
        for name in VARIANTS
    }
    paired_medians = {
        name: {clock: statistics.median(values) for clock, values in clocks.items()}
        for name, clocks in ratios.items()
    }
    wins = {
        name: {clock: sum(value > 1.0 for value in values) for clock, values in clocks.items()}
        for name, clocks in ratios.items()
    }
    return {
        "shape": {"vocab_size": vocab_size, "embedding_dim": embedding_dim},
        "ids_per_forward": len(representative_ids),
        "context_sample_count": len(id_batches),
        "unique_ids_per_context": {
            "min": min(len(set(ids)) for ids in id_batches),
            "max": max(len(set(ids)) for ids in id_batches),
            "mean": statistics.mean(len(set(ids)) for ids in id_batches),
        },
        "protocol": {
            "warmups_per_variant": warmups,
            "warmup_forward_calls_per_variant": warmups * len(warmup_ids),
            "warmup_order": warmup_order,
            "cycles_per_sample": cycles,
            "forward_calls_per_sample": cycles * len(id_batches),
            "block_orders": [list(order) for order in BLOCK_ORDERS],
            "timers": ["perf_counter_ns", "process_time_ns"],
            "ratio": "A_over_candidate = elapsed_A / elapsed_candidate; >1 favors candidate",
        },
        "one_call_exact_state_vs_A_on_clean_zero_gradients": equality,
        "output_and_input_copy_contract": copies,
        "single_forward_tracemalloc_bytes": allocations,
        "stale_gradient_two_batch_probe": stale_probe,
        "blocks": blocks,
        "median_ns_per_forward": medians,
        "paired_speedup_ratios": ratios,
        "median_paired_speedup": paired_medians,
        "candidate_wins_out_of_six_blocks": wins,
        "repeated_clean_forward_final_state_exact_by_block": repeated_state_exact,
    }


def benchmark(cycles: int, warmups: int, sample_count: int) -> dict[str, object]:
    char_batches, syllable_batches, input_stats = _actual_d_id_batches(sample_count)
    context_size = cast(int, input_stats["context_size"])
    embedding_dim = cast(int, input_stats["embedding_dim"])
    char_vocab_size = cast(int, input_stats["char_vocab_size"])
    syllable_vocab_size = cast(int, input_stats["syllable_vocab_size"])

    measurements = {
        "syllable_3932x8": _benchmark_shape(
            syllable_vocab_size,
            embedding_dim,
            syllable_batches,
            cycles=cycles,
            warmups=warmups,
            seed=7132,
        ),
        "character_51x8": _benchmark_shape(
            char_vocab_size,
            embedding_dim,
            char_batches,
            cycles=cycles,
            warmups=warmups,
            seed=7151,
        ),
    }
    stale = measurements["syllable_3932x8"]["stale_gradient_two_batch_probe"]
    assert isinstance(stale, dict)
    c_two_batch_exact = bool(stale["C"]["weights_exact_vs_A_after_two_batches"])
    exact_clean = True
    for shape in measurements.values():
        contracts = shape["one_call_exact_state_vs_A_on_clean_zero_gradients"]
        assert isinstance(contracts, dict)
        for flags in contracts.values():
            assert isinstance(flags, dict)
            exact_clean = exact_clean and all(bool(value) for value in flags.values())
    return {
        "phase": "7.1_embedding_forward_microbenchmark",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "input_source": input_stats,
        "candidate_definitions": {
            "A": "current production Embedding.forward()",
            "B": "cache vocab_size/embedding_dim locally; preserve all validation, output copies, and full zero-gradient matrix reset using [0.0] * embedding_dim rows",
            "C": "cache dimensions; preserve outputs/copies; reset only unique rows named by this forward call",
        },
        "measurements": measurements,
        "clean_state_all_candidates_exact": exact_clean,
        "C_preserves_full_reset_contract_across_two_batches": c_two_batch_exact,
        "production_code_changed": False,
        "decision": "microbenchmark_only_review_C_stale_gradient_failure_before_any_D_gate",
        "notes": [
            "Timing reuses actual D input contexts from the first train records; every candidate sees the same ordered contexts.",
            "C is timed on a clean zero-gradient state for forward latency, then separately probed with stale gradients from a prior batch.",
            "Tracemalloc is an isolated Python-allocation probe, not process RSS.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/phase7_1_embedding_forward_microbenchmark.json",
    )
    args = parser.parse_args()
    if args.cycles <= 0 or args.warmups < 0 or args.sample_count <= 0:
        parser.error("cycles/sample-count devem ser positivos e warmups não pode ser negativo")
    report = benchmark(args.cycles, args.warmups, args.sample_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
