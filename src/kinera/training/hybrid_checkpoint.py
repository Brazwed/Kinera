"""Versioned, CPU-only persistence for Kinera's complete hybrid model.

This is deliberately separate from the historical kinera.checkpoint v1
Neuron/Linear contract. R2.2 supports inference restoration only; the
training-resume contract belongs to R2.4.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from kinera.backends.cpu.backend import CPUBackend
from kinera.experiments.fingerprint import canonical_sha256
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.monitor.metrics.schema import METRICS_SCHEMA_VERSION, metric_result
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



# Epoch rows follow the two canonical trainers, not arbitrary JSON metadata.
# The multitask evaluator emits numeric scalars plus structured metrics,
# including legitimate null values for unavailable observations.
_MULTITASK_SCALARS = frozenset({
    "char_loss", "syllable_loss", "boundary_loss", "total_loss",
    "char_accuracy", "syllable_accuracy", "boundary_accuracy",
    "syllable_accuracy_all", "syllable_accuracy_known",
    "boundary_precision", "boundary_recall", "boundary_f1",
    "boundary_true_negative", "boundary_specificity",
    "boundary_balanced_accuracy", "boundary_baseline_constant_0",
    "boundary_baseline_constant_1", "boundary_baseline_majority",
})
_NULLABLE_MULTITASK_SCALARS = frozenset({
    "boundary_specificity", "boundary_balanced_accuracy",
    "boundary_baseline_majority",
})
_MULTITASK_DISTRIBUTIONS = frozenset({
    "boundary_target_distribution", "boundary_prediction_distribution",
    "char_target_distribution", "char_prediction_distribution",
    "syllable_target_distribution", "syllable_prediction_distribution",
    "syllable_target_status_distribution",
})


def _history_counts(
    value: Any,
    name: str,
    *,
    keys: set[str] | None = None,
    numeric_keys: bool = False,
) -> None:
    if not isinstance(value, dict) or (keys is not None and set(value) != keys):
        raise CheckpointError(f"{name}: distribuição inválida")
    for key, count in value.items():
        if type(key) is not str or not key:
            raise CheckpointError(f"{name}: chave inválida")
        if numeric_keys and re.fullmatch(r"(0|[1-9][0-9]*)", key) is None:
            raise CheckpointError(f"{name}: ID de distribuição inválido")
        if type(count) is not int or count < 0:
            raise CheckpointError(f"{name}.{key}: contagem inválida")


def _validate_history_row(value: Any, mode: str, index: int) -> None:
    name = f"training.history[{index}]"
    if mode == "char":
        row = _required(value, {"loss", "accuracy"}, name)
        if _number(row["loss"], f"{name}.loss") < 0:
            raise CheckpointError(f"{name}.loss deve ser não negativa")
        accuracy = _number(row["accuracy"], f"{name}.accuracy")
        if not 0 <= accuracy <= 1:
            raise CheckpointError(f"{name}.accuracy fora de [0,1]")
        return

    expected = (
        set(_MULTITASK_SCALARS) | set(_MULTITASK_DISTRIBUTIONS)
        | {"boundary_confusion", "metrics_schema_version", "metric_results"}
    )
    row = _required(value, expected, name)
    for field in _MULTITASK_SCALARS:
        item = row[field]
        if item is None and field in _NULLABLE_MULTITASK_SCALARS:
            continue
        number = _number(item, f"{name}.{field}")
        if field.endswith("_loss") and number < 0:
            raise CheckpointError(f"{name}.{field} deve ser não negativa")
    if type(row["boundary_true_negative"]) is not int or row["boundary_true_negative"] < 0:
        raise CheckpointError(f"{name}.boundary_true_negative inválido")
    if type(row["metrics_schema_version"]) is not int or row["metrics_schema_version"] != METRICS_SCHEMA_VERSION:
        raise CheckpointError(f"{name}.metrics_schema_version não suportada")

    _history_counts(
        row["boundary_confusion"], f"{name}.boundary_confusion",
        keys={"tp", "tn", "fp", "fn"},
    )
    for field in _MULTITASK_DISTRIBUTIONS:
        keys = None
        numeric_keys = True
        if field in ("boundary_target_distribution", "boundary_prediction_distribution"):
            keys = {"0", "1"}
        elif field == "syllable_target_status_distribution":
            keys = {"known", "unknown", "end"}
            numeric_keys = False
        _history_counts(
            row[field], f"{name}.{field}",
            keys=keys, numeric_keys=numeric_keys,
        )

    results = _required(
        row["metric_results"], set(_MULTITASK_SCALARS),
        f"{name}.metric_results",
    )
    for field, descriptor in results.items():
        location = f"{name}.metric_results.{field}"
        entry = _required(
            descriptor, {"value", "availability", "reason", "support"},
            location,
        )
        if not isinstance(entry["support"], dict):
            raise CheckpointError(f"{location}.support inválido")
        try:
            metric_result(
                entry["value"],
                availability=entry["availability"],
                reason=entry["reason"],
                support=entry["support"],
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise CheckpointError(f"{location}: descritor inválido: {error}") from error


def _validate_training_state(value: Any) -> dict[str, Any]:
    """Validate the versioned epoch-boundary resume record without trusting metadata."""
    try:
        state = _required(value, {
            "schema_version", "mode", "completed_epochs", "global_step",
            "selected_examples", "examples_sha256", "model_parameters_sha256",
            "configuration", "history",
        }, "training")
    except CheckpointError as error:
        # Preserve the R2.3 diagnostic for unsupported/partial resume records.
        raise CheckpointError(f"estado de treinamento R2.4 inválido: {error}") from error
    if type(state["schema_version"]) is not int or state["schema_version"] != 1:
        raise CheckpointError("versão do estado de treinamento não suportada")
    if state["mode"] not in ("char", "multitask"):
        raise CheckpointError("modo do treinamento inválido")
    for key in ("completed_epochs", "global_step"):
        if type(state[key]) is not int or state[key] < 0:
            raise CheckpointError(f"training.{key} deve ser inteiro não negativo")
    _dimension(state["selected_examples"], "training.selected_examples")
    if state["global_step"] != state["completed_epochs"] * state["selected_examples"]:
        raise CheckpointError("training.global_step incompatível com épocas completas")
    for field in ("examples_sha256", "model_parameters_sha256"):
        if (
            type(state[field]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", state[field]) is None
        ):
            raise CheckpointError(f"training.{field} inválido")
    config = _required(state["configuration"], {
        "learning_rate", "max_examples", "syllable_unknown_id", "loss_weights",
    }, "training.configuration")
    if _number(config["learning_rate"], "learning_rate") <= 0:
        raise CheckpointError("learning_rate deve ser positivo")
    max_examples = config["max_examples"]
    if max_examples is not None:
        _dimension(max_examples, "max_examples")
    unknown = config["syllable_unknown_id"]
    if type(unknown) is not int or unknown < 0:
        raise CheckpointError("syllable_unknown_id deve ser inteiro não negativo")
    weights = config["loss_weights"]
    if state["mode"] == "char":
        if weights is not None:
            raise CheckpointError("loss_weights deve ser null em modo char")
    else:
        weights = _required(weights, {"char", "syllable", "boundary"}, "loss_weights")
        normalized = [_number(value, f"loss_weights.{name}") for name, value in weights.items()]
        if any(weight < 0 for weight in normalized) or not any(weight > 0 for weight in normalized):
            raise CheckpointError("loss_weights inválidos")
    history = state["history"]
    if not isinstance(history, list) or len(history) != state["completed_epochs"]:
        raise CheckpointError("training.history incompatível com épocas completas")
    for index, row in enumerate(history):
        _validate_history_row(row, state["mode"], index)
    _ensure_json_compatible(state)
    return state


def save_autoregressive_checkpoint(
    path: str | Path,
    model: AutoregressiveModel,
    *,
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    metadata: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically persist CPU weights and optionally versioned epoch-boundary state."""
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
        "training": None if training_state is None else _validate_training_state(training_state),
        "metadata": {} if metadata is None else metadata.copy(),
    }
    _ensure_json_compatible(payload)
    return _write_payload(path, payload)


def _parse_constant(value: str) -> Any:
    raise CheckpointError(f"constante JSON inválida: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous keys at every JSON object nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"chave JSON duplicada: {key}")
        result[key] = value
    return result


def _read_hybrid(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_parse_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"não foi possível ler checkpoint híbrido: {exc}") from exc
    # Distinguish legacy/wrong formats before checking fields unique to R2.
    if isinstance(payload, dict) and payload.get("format") != HYBRID_CHECKPOINT_FORMAT:
        raise CheckpointError("formato de checkpoint híbrido inválido")
    _required(payload, {
        "format", "version", "model", "identity", "training", "metadata",
    }, "checkpoint")
    if type(payload["version"]) is not int or payload["version"] != HYBRID_CHECKPOINT_VERSION:
        raise CheckpointError("versão de checkpoint híbrido não suportada")
    if payload["training"] is not None:
        _validate_training_state(payload["training"])
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
    token-ID mappings. Validated training metadata is returned, never executed.
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

    return restored, {"training": payload["training"], "metadata": payload["metadata"].copy()}


__all__ = [
    "HYBRID_CHECKPOINT_FORMAT",
    "HYBRID_CHECKPOINT_VERSION",
    "save_autoregressive_checkpoint",
    "load_autoregressive_checkpoint",
]
