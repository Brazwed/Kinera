"""Persistência versionada de pesos para os modelos da Kinera."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from kinera.nn.layers.linear import Linear
from kinera.nn.layers.neuron import Neuron

CHECKPOINT_FORMAT = "kinera.checkpoint"
CHECKPOINT_VERSION = 1
Model = Neuron | Linear


class CheckpointError(Exception):
    """Erro de leitura, validação ou escrita de checkpoint."""


def save_neuron(
    path: str | Path,
    neuron: Neuron,
    *,
    training_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Salva um Neuron em JSON versionado."""
    if not isinstance(neuron, Neuron):
        raise TypeError("neuron deve ser uma instância de Neuron")
    payload = _base_payload(
        model={
            "type": "neuron",
            "input_size": len(neuron.weights),
            "activation": neuron.activation,
            "negative_slope": neuron.negative_slope,
            "weights": neuron.weights.copy(),
            "bias": neuron.bias,
        },
        training_state=training_state,
        metadata=metadata,
    )
    _validate_neuron_model(payload["model"])
    return _write_payload(path, payload)


def load_neuron(path: str | Path) -> tuple[Neuron, dict[str, Any]]:
    """Carrega um Neuron e retorna também training/metadata."""
    payload = _read_payload(path)
    model_data = payload["model"]
    if model_data.get("type") != "neuron":
        raise CheckpointError("checkpoint não contém um Neuron")
    _validate_neuron_model(model_data)
    try:
        neuron = Neuron(
            input_size=model_data["input_size"],
            activation=model_data["activation"],
            negative_slope=model_data["negative_slope"],
        )
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"arquitetura do Neuron inválida: {error}") from error
    neuron.weights = [float(value) for value in model_data["weights"]]
    neuron.bias = float(model_data["bias"])
    return neuron, _checkpoint_info(payload)


def save_linear(
    path: str | Path,
    layer: Linear,
    *,
    training_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Salva uma camada Linear em JSON versionado."""
    if not isinstance(layer, Linear):
        raise TypeError("layer deve ser uma instância de Linear")
    payload = _base_payload(
        model={
            "type": "linear",
            "input_size": len(layer.weights[0]),
            "output_size": len(layer.weights),
            "weights": [row.copy() for row in layer.weights],
            "bias": layer.bias.copy(),
        },
        training_state=training_state,
        metadata=metadata,
    )
    _validate_linear_model(payload["model"])
    return _write_payload(path, payload)


def load_linear(path: str | Path) -> tuple[Linear, dict[str, Any]]:
    """Carrega uma camada Linear e retorna também training/metadata."""
    payload = _read_payload(path)
    model_data = payload["model"]
    if model_data.get("type") != "linear":
        raise CheckpointError("checkpoint não contém uma camada Linear")
    _validate_linear_model(model_data)
    try:
        layer = Linear(
            input_size=model_data["input_size"],
            output_size=model_data["output_size"],
        )
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"arquitetura da Linear inválida: {error}") from error
    layer.weights = [row.copy() for row in model_data["weights"]]
    layer.bias = model_data["bias"].copy()
    return layer, _checkpoint_info(payload)


def save_checkpoint(
    path: str | Path,
    model: Model,
    *,
    training_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Salva automaticamente um Neuron ou Linear."""
    if isinstance(model, Neuron):
        return save_neuron(
            path,
            model,
            training_state=training_state,
            metadata=metadata,
        )
    if isinstance(model, Linear):
        return save_linear(
            path,
            model,
            training_state=training_state,
            metadata=metadata,
        )
    raise TypeError("model deve ser Neuron ou Linear")


def load_checkpoint(path: str | Path) -> tuple[Model, dict[str, Any]]:
    """Carrega automaticamente um Neuron ou Linear."""
    payload = _read_payload(path)
    model_type = payload["model"].get("type")
    if model_type == "neuron":
        return load_neuron(path)
    if model_type == "linear":
        return load_linear(path)
    raise CheckpointError(f"tipo de modelo desconhecido: {model_type}")


def _base_payload(
    *,
    model: dict[str, Any],
    training_state: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if training_state is not None and not isinstance(training_state, dict):
        raise TypeError("training_state deve ser um dicionário ou None")
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata deve ser um dicionário ou None")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model": model,
        "training": {} if training_state is None else training_state.copy(),
        "metadata": {} if metadata is None else metadata.copy(),
    }
    _ensure_json_compatible(payload)
    return payload


def _write_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except (OSError, TypeError, ValueError) as error:
        raise CheckpointError(f"não foi possível salvar checkpoint: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _read_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"não foi possível ler checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint deve conter um objeto JSON")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("formato de checkpoint inválido")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise CheckpointError("versão de checkpoint não suportada")
    model = payload.get("model")
    if not isinstance(model, dict):
        raise CheckpointError("bloco model ausente ou inválido")
    for key in ("training", "metadata"):
        if key not in payload:
            payload[key] = {}
        if not isinstance(payload[key], dict):
            raise CheckpointError(f"bloco {key} deve ser um objeto")
    return payload


def _checkpoint_info(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "training": payload["training"].copy(),
        "metadata": payload["metadata"].copy(),
    }


def _validate_neuron_model(model: Any) -> None:
    if not isinstance(model, dict):
        raise CheckpointError("bloco model inválido")
    required = {
        "type",
        "input_size",
        "activation",
        "negative_slope",
        "weights",
        "bias",
    }
    _require_keys(model, required)
    if model["type"] != "neuron":
        raise CheckpointError("tipo de modelo não é neuron")
    _validate_positive_int(model["input_size"], "input_size")
    if not isinstance(model["activation"], str):
        raise CheckpointError("activation inválida")
    _validate_finite_number(model["negative_slope"], "negative_slope")
    if not isinstance(model["weights"], list):
        raise CheckpointError("weights do Neuron devem ser uma lista")
    if len(model["weights"]) != model["input_size"]:
        raise CheckpointError("quantidade de weights incompatível")
    for index, value in enumerate(model["weights"]):
        _validate_finite_number(value, f"weights[{index}]")
    _validate_finite_number(model["bias"], "bias")


def _validate_linear_model(model: Any) -> None:
    if not isinstance(model, dict):
        raise CheckpointError("bloco model inválido")
    required = {"type", "input_size", "output_size", "weights", "bias"}
    _require_keys(model, required)
    if model["type"] != "linear":
        raise CheckpointError("tipo de modelo não é linear")
    _validate_positive_int(model["input_size"], "input_size")
    _validate_positive_int(model["output_size"], "output_size")
    weights = model["weights"]
    bias = model["bias"]
    if not isinstance(weights, list) or len(weights) != model["output_size"]:
        raise CheckpointError("shape das weights da Linear inválido")
    if not isinstance(bias, list) or len(bias) != model["output_size"]:
        raise CheckpointError("shape do bias da Linear inválido")
    for row_index, row in enumerate(weights):
        if not isinstance(row, list) or len(row) != model["input_size"]:
            raise CheckpointError(f"shape de weights[{row_index}] inválido")
        for column_index, value in enumerate(row):
            _validate_finite_number(
                value,
                f"weights[{row_index}][{column_index}]",
            )
    for index, value in enumerate(bias):
        _validate_finite_number(value, f"bias[{index}]")


def _require_keys(mapping: dict[str, Any], keys: set[str]) -> None:
    missing = keys.difference(mapping)
    if missing:
        raise CheckpointError(f"campos ausentes no model: {sorted(missing)}")


def _validate_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointError(f"{name} deve ser inteiro")
    if value <= 0:
        raise CheckpointError(f"{name} deve ser positivo")


def _validate_finite_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(f"{name} deve ser numérico")
    if not math.isfinite(value):
        raise CheckpointError(f"{name} deve ser finito")


def _ensure_json_compatible(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CheckpointError(f"valor não finito em {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise CheckpointError(f"chave não textual em {path}")
            _ensure_json_compatible(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _ensure_json_compatible(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise CheckpointError(f"valor não serializável em {path}")
