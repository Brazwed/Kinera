"""Smoke test de cinco minutos para a representação híbrida da Kinera."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from language.hybrid_inputs import record_to_hybrid_inputs
from language.vocabulary import CharacterVocabulary, SyllableVocabulary
from nn.embedding import Embedding
from nn.hybrid import HybridRepresentation
from nn.linear import Linear

DURATION_SECONDS = 300.0
REPORT_INTERVAL_SECONDS = 30.0
LEARNING_RATE_HEAD = 0.01
LEARNING_RATE_EMBEDDINGS = 0.005
TRAIN_RECORD_LIMIT = 400
VALIDATION_RECORD_LIMIT = 100
CHAR_EMBEDDING_DIM = 8
SYLLABLE_EMBEDDING_DIM = 8
SEED = 7
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "processed"
LOG_PATH = ROOT / "hybrid_5min.log"


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= limit:
                    break
    return records


def build_samples(
    records: list[dict[str, Any]],
    char_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
) -> list[tuple[int, int, list[float], int]]:
    samples = []
    for record in records:
        char_ids, syllable_ids, feature_vectors = record_to_hybrid_inputs(
            record, char_vocabulary, syllable_vocabulary
        )
        characters = record["characters"]
        for index in range(len(char_ids)):
            if index + 1 < len(characters):
                target_id = char_vocabulary.encode(characters[index + 1]["char"])[0]
            else:
                target_id = char_vocabulary.end_id
            samples.append(
                (char_ids[index], syllable_ids[index], feature_vectors[index], target_id)
            )
    if not samples:
        raise ValueError("nenhuma amostra autoregressiva foi gerada")
    return samples


def metrics(
    samples: list[tuple[int, int, list[float], int]],
    representation: HybridRepresentation,
    head: Linear,
    char_embedding: Embedding,
    syllable_embedding: Embedding,
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    for char_id, syllable_id, features, target_id in samples:
        vector = representation.forward([char_id], [syllable_id], [features])[0]
        prediction = head.forward(vector)
        target = [0.0 for _ in prediction]
        target[target_id] = 1.0
        total_loss += 0.5 * sum((value - expected) ** 2 for value, expected in zip(prediction, target))
        if max(range(len(prediction)), key=prediction.__getitem__) == target_id:
            correct += 1
    return total_loss / len(samples), correct / len(samples)


def main() -> None:
    char_vocabulary = CharacterVocabulary.from_dict(
        json.loads((DATA_ROOT / "vocabulary.json").read_text(encoding="utf-8"))
    )
    syllable_vocabulary = SyllableVocabulary.from_dict(
        json.loads(
            (DATA_ROOT / "syllable_vocabulary.json").read_text(encoding="utf-8")
        )
    )
    train_records = load_records(
        DATA_ROOT / "annotated_syllable_records.jsonl", TRAIN_RECORD_LIMIT
    )
    validation_records = load_records(
        DATA_ROOT / "annotated_syllable_records.jsonl",
        TRAIN_RECORD_LIMIT + VALIDATION_RECORD_LIMIT,
    )[TRAIN_RECORD_LIMIT:]
    train_samples = build_samples(train_records, char_vocabulary, syllable_vocabulary)
    validation_samples = build_samples(
        validation_records, char_vocabulary, syllable_vocabulary
    )

    char_embedding = Embedding(len(char_vocabulary), CHAR_EMBEDDING_DIM, seed=SEED)
    syllable_embedding = Embedding(
        len(syllable_vocabulary), SYLLABLE_EMBEDDING_DIM, seed=SEED + 1
    )
    representation = HybridRepresentation(
        char_embedding,
        syllable_embedding,
        feature_dim=10,
    )
    head = Linear(representation.output_dim, len(char_vocabulary), seed=SEED + 2)
    rng = random.Random(SEED)

    initial_loss, initial_accuracy = metrics(
        validation_samples,
        representation,
        head,
        char_embedding,
        syllable_embedding,
    )
    started_at = time.monotonic()
    deadline = started_at + DURATION_SECONDS
    next_report = started_at + REPORT_INTERVAL_SECONDS
    steps = 0
    epochs = 0

    with LOG_PATH.open("w", encoding="utf-8") as log:
        header = (
            "Kinera — treino híbrido de 5 minutos\n"
            f"train_records={len(train_records)} train_samples={len(train_samples)}\n"
            f"validation_records={len(validation_records)} validation_samples={len(validation_samples)}\n"
            f"representation_dim={representation.output_dim} vocabulary_size={len(char_vocabulary)}\n"
            f"learning_rate_head={LEARNING_RATE_HEAD} learning_rate_embeddings={LEARNING_RATE_EMBEDDINGS}\n"
            f"initial_validation_loss={initial_loss:.12f} initial_validation_accuracy={initial_accuracy:.6f}\n"
        )
        print(header, end="", flush=True)
        log.write(header)
        log.flush()

        while time.monotonic() < deadline:
            rng.shuffle(train_samples)
            epochs += 1
            for char_id, syllable_id, features, target_id in train_samples:
                if time.monotonic() >= deadline:
                    break
                vector = representation.forward([char_id], [syllable_id], [features])[0]
                prediction = head.forward(vector)
                target = [0.0 for _ in prediction]
                target[target_id] = 1.0
                error = [value - expected for value, expected in zip(prediction, target)]
                grad_vector = head.backward(error)
                representation.backward([grad_vector])
                head.step(LEARNING_RATE_HEAD)
                representation.step(LEARNING_RATE_EMBEDDINGS)
                steps += 1

                now = time.monotonic()
                if now >= next_report:
                    validation_loss, validation_accuracy = metrics(
                        validation_samples,
                        representation,
                        head,
                        char_embedding,
                        syllable_embedding,
                    )
                    message = (
                        f"elapsed={now - started_at:.1f}s epochs={epochs} steps={steps} "
                        f"validation_loss={validation_loss:.12f} "
                        f"validation_accuracy={validation_accuracy:.6f}"
                    )
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()
                    next_report += REPORT_INTERVAL_SECONDS

        final_loss, final_accuracy = metrics(
            validation_samples,
            representation,
            head,
            char_embedding,
            syllable_embedding,
        )
        summary = (
            f"finished elapsed={time.monotonic() - started_at:.3f}s epochs={epochs} steps={steps} "
            f"initial_loss={initial_loss:.12f} final_loss={final_loss:.12f} "
            f"initial_accuracy={initial_accuracy:.6f} final_accuracy={final_accuracy:.6f}"
        )
        print(summary, flush=True)
        log.write(summary + "\n")


if __name__ == "__main__":
    main()
