"""Versioned, CPU-only persistence for Kinera's complete hybrid model.

This is deliberately separate from the historical kinera.checkpoint v1
Neuron/Linear contract. R2.2 supports inference restoration only; the
training-resume contract belongs to R2.4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from kinera.backends.cpu.backend import CPUBackend
from kinera.experiments.fingerprint import canonical_sha256
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.checkpoint import (
    CheckpointError,
    _ensure_json_compatible,
    _write_payload,
)

HYBRID_CHECKPOINT_FORMAT = "kinera.hybrid.checkpoint"
HYBRID_CHECKPOINT_VERSION = 1


def _required(mapping: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or set(mapping) != keys:
        raise CheckpointError(f"{location}: campos ausentes ou desconhecidos")
    return mapping


def _dimension(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise CheckpointError(f"{name} deve ser inteiro positivo")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise CheckpointError(f"{name} deve ser numérico, não booleano")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise CheckpointError(f"{name} inválido") from exc
    if not math.isfinite(result):
        raise CheckpointError(f"{name} deve ser finito")
    return result


def _vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise CheckpointError(f"{name}: shape inválido")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _matrix(value: Any, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise CheckpointError(f"{name}: shape inválido")
    return [
        _vector(row, columns, f"{name}[{index}]")
        for index, row in enumerate(value)
    ]


def _vocabularies(
    character: CharacterVocabulary,
    syllable: SyllableVocabulary,
    *,
    char_size: int,
    syllable_size: int,
) -> None:
    if not isinstance(character, CharacterVocabulary):
        raise TypeError("character_vocabulary deve ser CharacterVocabulary")
    if not isinstance(syllable, SyllableVocabulary):
        raise TypeError("syllable_vocabulary deve ser SyllableVocabulary")
    if len(character) != char_size or len(syllable) != syllable_size:
        raise CheckpointError("tamanhos dos vocabulários incompatíveis com o modelo")


def _identity(
    character: CharacterVocabulary, syllable: SyllableVocabulary
) -> dict[str, Any]:
    char_mapping = character.to_dict()
    syl_mapping = syllable.to_dict()
    return {
        "algorithm": "sha256",
        "character_vocabulary": char_mapping,
        "syllable_vocabulary": syl_mapping,
        "character_sha256": canonical_sha256({
            "mapping": char_mapping,
            "special_ids": {
                "start": character.start_id,
                "end": character.end_id,
                "unknown": character.unk_id,
            },
        }),
        "syllable_sha256": canonical_sha256({
            "mapping": syl_mapping,
            "unknown_id": syllable.unk_id,
        }),
    }


def _architecture(model: AutoregressiveModel) -> dict[str, int]:
    representation = model.representation
    char = representation.char_embedding
    syl = representation.syllable_embedding
    return {
        "char_vocab_size": char.vocab_size,
        "syllable_vocab_size": syl.vocab_size,
        "char_embedding_dim": char.embedding_dim,
        "syllable_embedding_dim": syl.embedding_dim,
        "feature_dim": representation.feature_dim,
    }


def _parameters(model: AutoregressiveModel, arch: dict[str, int]) -> dict[str, Any]:
    features = arch["feature_dim"]
    input_dim = arch["char_embedding_dim"] + features + arch["syllable_embedding_dim"]
    result: dict[str, Any] = {
        "char_embedding": {
            "weights": _matrix(
                model.representation.char_embedding.weights,
                arch["char_vocab_size"], arch["char_embedding_dim"],
                "char_embedding.weights",
            ),
        },
        "syllable_embedding": {
            "weights": _matrix(
                model.representation.syllable_embedding.weights,
                arch["syllable_vocab_size"], arch["syllable_embedding_dim"],
                "syllable_embedding.weights",
            ),
        },
        "heads": {},
    }
    for name, rows in (
        ("char", arch["char_vocab_size"]),
        ("syllable", arch["syllable_vocab_size"]),
        ("boundary", 1),
    ):
        layer = getattr(model.heads, name)
        if layer.input_size != input_dim or layer.output_size != rows:
            raise CheckpointError(f"heads.{name}: arquitetura inconsistente")
        result["heads"][name] = {
            "weights": _matrix(layer.weights, rows, input_dim, f"heads.{name}.weights"),
            "bias": _vector(layer.bias, rows, f"heads.{name}.bias"),
        }
    return result


def save_autoregressive_checkpoint(
    path: str | Path,
    model: AutoregressiveModel,
    *,
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save all CPU model parameters for later inference."""
    if not isinstance(model, AutoregressiveModel):
        raise TypeError("model deve ser AutoregressiveModel")
    if not isinstance(model.backend, CPUBackend):
        raise CheckpointError("checkpoint híbrido R2.2 suporta apenas CPUBackend")
    if model._closed:
        raise CheckpointError("não é possível salvar um modelo fechado")
    arch = _architecture(model)
    _vocabularies(
        character_vocabulary, syllable_vocabulary,
        char_size=arch["char_vocab_size"],
        syllable_size=arch["syllable_vocab_size"],
    )
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata deve ser dicionário ou None")
    payload = {
        "format": HYBRID_CHECKPOINT_FORMAT,
        "version": HYBRID_CHECKPOINT_VERSION,
        "model": {
            "type": "autoregressive_hybrid",
            "architecture": arch,
            "parameters": _parameters(model, arch),
        },
        "identity": _identity(character_vocabulary, syllable_vocabulary),
        "training": None,
        "metadata": {} if metadata is None else metadata.copy(),
    }
    _ensure_json_compatible(payload)
    return _write_payload(path, payload)


def _parse_constant(value: str) -> Any:
    raise CheckpointError(f"constante JSON inválida: {value}")


def _read_hybrid(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_parse_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"não foi possível ler checkpoint híbrido: {exc}") from exc
    _required(payload, {
        "format", "version", "model", "identity", "training", "metadata",
    }, "checkpoint")
    if payload["format"] != HYBRID_CHECKPOINT_FORMAT:
        raise CheckpointError("formato de checkpoint híbrido inválido")
    if type(payload["version"]) is not int or payload["version"] != HYBRID_CHECKPOINT_VERSION:
        raise CheckpointError("versão de checkpoint híbrido não suportada")
    if payload["training"] is not None:
        raise CheckpointError("estado de retomada ainda não é suportado (R2.4)")
    if not isinstance(payload["metadata"], dict):
        raise CheckpointError("metadata deve ser dicionário")
    _ensure_json_compatible(payload["metadata"])
    return payload


def load_autoregressive_checkpoint(
    path: str | Path,
    *,
    expected_character_vocabulary: CharacterVocabulary,
    expected_syllable_vocabulary: SyllableVocabulary,
) -> tuple[AutoregressiveModel, dict[str, Any]]:
    """Validate fully, then return a NEW CPU model with clean transient state.

    Both expected vocabularies are mandatory: equal sizes do not imply equal
    token-ID mappings. Does not resume training.
    """
    payload = _read_hybrid(path)
    model_payload = _required(
        payload["model"], {"type", "architecture", "parameters"}, "model"
    )
    if model_payload["type"] != "autoregressive_hybrid":
        raise CheckpointError("tipo de modelo híbrido desconhecido")

    arch = _required(model_payload["architecture"], {
        "char_vocab_size", "syllable_vocab_size",
        "char_embedding_dim", "syllable_embedding_dim", "feature_dim",
    }, "architecture")
    dims = {key: _dimension(value, key) for key, value in arch.items()}
    _vocabularies(
        expected_character_vocabulary, expected_syllable_vocabulary,
        char_size=dims["char_vocab_size"],
        syllable_size=dims["syllable_vocab_size"],
    )
    identity = _required(payload["identity"], {
        "algorithm", "character_vocabulary", "syllable_vocabulary",
        "character_sha256", "syllable_sha256",
    }, "identity")
    if identity["algorithm"] != "sha256":
        raise CheckpointError("algoritmo de identidade não suportado")
    char_data = identity["character_vocabulary"]
    syl_data = identity["syllable_vocabulary"]
    try:
        char_vocabulary = CharacterVocabulary.from_dict(char_data)
        syl_vocabulary = SyllableVocabulary.from_dict(syl_data)
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointError(f"mapeamento de vocabulário inválido: {exc}") from exc
    if identity != _identity(char_vocabulary, syl_vocabulary):
        raise CheckpointError("digest ou identidade de vocabulário inválidos")
    if (
        char_vocabulary.to_dict() != expected_character_vocabulary.to_dict()
        or syl_vocabulary.to_dict() != expected_syllable_vocabulary.to_dict()
    ):
        raise CheckpointError("identidade de vocabulário incompatível")

    params = _required(model_payload["parameters"], {
        "char_embedding", "syllable_embedding", "heads",
    }, "parameters")
    c = _required(params["char_embedding"], {"weights"}, "char_embedding")
    s = _required(params["syllable_embedding"], {"weights"}, "syllable_embedding")
    inputs = dims["char_embedding_dim"] + dims["feature_dim"] + dims["syllable_embedding_dim"]
    char_weights = _matrix(
        c["weights"], dims["char_vocab_size"], dims["char_embedding_dim"], "char_embedding.weights"
    )
    syl_weights = _matrix(
        s["weights"], dims["syllable_vocab_size"], dims["syllable_embedding_dim"],
        "syllable_embedding.weights",
    )
    heads = _required(params["heads"], {"char", "syllable", "boundary"}, "heads")
    restored_heads: dict[str, dict[str, Any]] = {}
    for name, count in (
        ("char", dims["char_vocab_size"]),
        ("syllable", dims["syllable_vocab_size"]),
        ("boundary", 1),
    ):
        head = _required(heads[name], {"weights", "bias"}, f"heads.{name}")
        restored_heads[name] = {
            "weights": _matrix(head["weights"], count, inputs, f"heads.{name}.weights"),
            "bias": _vector(head["bias"], count, f"heads.{name}.bias"),
        }

    # All externally supplied content is validated before allocating or
    # modifying a model. A failed load never alters a caller-owned instance.
    restored = AutoregressiveModel(
        char_vocab_size=dims["char_vocab_size"],
        syllable_vocab_size=dims["syllable_vocab_size"],
        char_embedding_dim=dims["char_embedding_dim"],
        syllable_embedding_dim=dims["syllable_embedding_dim"],
        feature_dim=dims["feature_dim"],
        backend=CPUBackend(),
    )
    restored.representation.char_embedding.weights = char_weights
    restored.representation.syllable_embedding.weights = syl_weights
    for name in ("char", "syllable", "boundary"):
        layer = getattr(restored.heads, name)
        layer.weights = restored_heads[name]["weights"]
        layer.bias = restored_heads[name]["bias"]

    return restored, {"training": None, "metadata": payload["metadata"].copy()}


__all__ = [
    "HYBRID_CHECKPOINT_FORMAT",
    "HYBRID_CHECKPOINT_VERSION",
    "save_autoregressive_checkpoint",
    "load_autoregressive_checkpoint",
]
