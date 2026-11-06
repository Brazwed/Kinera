"""Stable, type-preserving hashes for scientific experiment inputs."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from kinera.language.datasets.loader import load_jsonl
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary


FINGERPRINT_SCHEMA_VERSION = 1


def _write_length(digest: Any, length: int) -> None:
    digest.update(length.to_bytes(8, byteorder="big", signed=False))


def _update_canonical(digest: Any, value: Any) -> None:
    """Stream a typed canonical representation into ``digest``.

    Supported values mirror JSON data used by the experiment runner. Mapping
    keys are strings and sorted by Unicode code point; list order is retained.
    Floats use their exact hexadecimal representation, not locale or repr().
    """
    value_type = type(value)
    if value is None:
        digest.update(b"n")
    elif value_type is bool:
        digest.update(b"b1" if value else b"b0")
    elif value_type is int:
        payload = str(value).encode("ascii")
        digest.update(b"i")
        _write_length(digest, len(payload))
        digest.update(payload)
    elif value_type is float:
        if not math.isfinite(value):
            raise ValueError("fingerprint não aceita float NaN ou infinito")
        payload = value.hex().encode("ascii")
        digest.update(b"f")
        _write_length(digest, len(payload))
        digest.update(payload)
    elif value_type is str:
        payload = value.encode("utf-8")
        digest.update(b"s")
        _write_length(digest, len(payload))
        digest.update(payload)
    elif value_type is list:
        digest.update(b"l")
        _write_length(digest, len(value))
        for item in value:
            _update_canonical(digest, item)
    elif value_type is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("fingerprint exige chaves de dicionário string")
        keys = sorted(value)
        digest.update(b"d")
        _write_length(digest, len(keys))
        for key in keys:
            _update_canonical(digest, key)
            _update_canonical(digest, value[key])
    else:
        raise TypeError(
            f"tipo não suportado no fingerprint: {value_type.__name__}"
        )


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 of a deterministic, typed encoding of JSON-like data."""
    digest = hashlib.sha256()
    _update_canonical(digest, value)
    return digest.hexdigest()


def fingerprint_components_digest(
    components: dict[str, str],
    *,
    schema_version: int = FINGERPRINT_SCHEMA_VERSION,
) -> str:
    """Hash component digests using the versioned experiment fingerprint envelope."""
    return canonical_sha256({
        "fingerprint_schema_version": schema_version,
        "algorithm": "sha256",
        "components": components,
    })


def validate_fingerprint_manifest(
    persisted: Any,
    expected: dict[str, Any],
    *,
    component: str,
) -> None:
    """Reject legacy, malformed, unknown-version, or different identities."""
    if not isinstance(persisted, dict):
        raise ValueError(f"{component}: checkpoint legado sem identidade científica")
    if (
        type(persisted.get("fingerprint_schema_version")) is not int
        or persisted.get("fingerprint_schema_version") != FINGERPRINT_SCHEMA_VERSION
        or persisted.get("algorithm") != "sha256"
        or persisted.get("phase") != expected.get("phase")
    ):
        raise ValueError(f"{component}: identidade científica ausente ou incompatível")
    components = persisted.get("components")
    expected_components = expected.get("components")
    if (
        not isinstance(components, dict)
        or not isinstance(expected_components, dict)
        or set(components) != set(expected_components)
        or any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in components.values()
        )
    ):
        raise ValueError(f"{component}: componentes do fingerprint inválidos")
    calculated_digest = fingerprint_components_digest(components)
    if persisted.get("digest") != calculated_digest:
        raise ValueError(f"{component}: digest interno do fingerprint inválido")
    changed = [
        name
        for name, digest in expected_components.items()
        if components.get(name) != digest
    ]
    if changed:
        raise ValueError(
            f"{component}: identidade científica incompatível: "
            + ", ".join(changed)
        )
    if persisted != expected:
        raise ValueError(f"{component}: manifesto de identidade incompatível")


def validate_legacy_checkpoint_identity(
    checkpoint_path: str,
    expected: dict[str, Any],
    *,
    phase: str,
) -> None:
    """Validate checkpoint identity read-only before the generic runner writes."""
    import json
    from pathlib import Path

    source = Path(checkpoint_path)
    if not source.exists():
        raise ValueError("checkpoint inexistente")
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint inválido") from exc
    if not isinstance(report, dict):
        raise ValueError("checkpoint inválido")
    validate_fingerprint_manifest(
        report.get("experiment_identity"),
        expected,
        component=phase,
    )


def build_legacy_experiment_fingerprint(
    *,
    phase: str,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    scientific_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Build the M1.4-compatible identity envelope used by legacy seed runners.

    Dataset records are hashed in full and in order. This intentionally may
    invalidate a resume when unused record metadata changes; paths and runtime
    settings are excluded by the caller from ``scientific_configuration``.
    """
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase deve ser uma string não vazia")
    if not isinstance(scientific_configuration, dict):
        raise TypeError("scientific_configuration deve ser um dicionário")
    components = {
        "training_examples_sha256": canonical_sha256(train_examples),
        "validation_examples_sha256": canonical_sha256(validation_examples),
        "character_vocabulary_sha256": canonical_sha256({
            "mapping": vocabulary.to_dict(),
            "special_ids": {
                "start": vocabulary.start_id,
                "end": vocabulary.end_id,
                "unknown": vocabulary.unk_id,
            },
        }),
        "syllable_vocabulary_sha256": canonical_sha256({
            "mapping": syllable_vocabulary.to_dict(),
            "unknown_id": syllable_vocabulary.unk_id,
        }),
        "scientific_configuration_sha256": canonical_sha256({
            "phase": phase,
            "configuration": scientific_configuration,
        }),
    }
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "phase": phase,
        "components": components,
        "digest": fingerprint_components_digest(components),
    }


def build_legacy_scientific_configuration(
    experiment_configuration: dict[str, Any],
    *,
    vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
) -> dict[str, Any]:
    """Return science-affecting runner settings, excluding seeds and operations."""
    configuration = dict(experiment_configuration)
    configuration.pop("seeds", None)
    configuration.update({
        "phase_protocol_version": 1,
        "character_vocab_size": len(vocabulary),
        "syllable_vocab_size": len(syllable_vocabulary),
        "syllable_unknown_id": syllable_vocabulary.unk_id,
        "model": {
            "class": "AutoregressiveModel",
            "char_embedding_dim": 8,
            "syllable_embedding_dim": 8,
            "feature_dim": 10,
        },
        "loss_weights": {"char": 1.0, "syllable": 0.7, "boundary": 0.3},
        "training_algorithm": "multitask_sgd",
    })
    return configuration


def load_legacy_experiment_inputs(
    payload: dict[str, Any],
    *,
    phase: str,
    require_expected_identity: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    CharacterVocabulary,
    SyllableVocabulary,
    dict[str, Any],
]:
    """Load effective legacy-runner inputs and verify the worker's identity.

    The same loader and prefix rule are used both before checkpoint resume and
    inside each worker, guarding against files changing after preflight.
    """
    import json
    from pathlib import Path

    vocabulary = CharacterVocabulary.from_dict(
        json.loads(Path(payload["vocabulary"]).read_text(encoding="utf-8"))
    )
    syllable_vocabulary = SyllableVocabulary.from_dict(
        json.loads(Path(payload["syllable_vocabulary"]).read_text(encoding="utf-8"))
    )
    max_examples = payload["max_examples"]
    train_examples = load_jsonl(payload["train"], max_records=max_examples)
    validation_examples = load_jsonl(
        payload["validation"], max_records=max_examples
    )
    identity = build_legacy_experiment_fingerprint(
        phase=phase,
        train_examples=train_examples,
        validation_examples=validation_examples,
        vocabulary=vocabulary,
        syllable_vocabulary=syllable_vocabulary,
        scientific_configuration=payload["scientific_configuration"],
    )
    expected = payload.get("experiment_identity")
    if require_expected_identity and expected is None:
        raise ValueError("identidade científica ausente no payload do worker")
    if expected is not None:
        validate_fingerprint_manifest(
            expected,
            identity,
            component=f"{phase} worker input",
        )
    return (
        train_examples,
        validation_examples,
        vocabulary,
        syllable_vocabulary,
        identity,
    )
