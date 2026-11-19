"""R2.4: exact epoch-boundary CPU resume for the existing hybrid training loops.

The canonical trainers are called for one complete epoch at a time. No changes
to their mathematics, dataset adapters, CLI, observers, or scientific runners.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from kinera.backends.cpu.backend import CPUBackend
from kinera.experiments.fingerprint import canonical_sha256
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.losses import MultiTaskLoss
from kinera.nn.models.hybrid import AutoregressiveModel
from kinera.training.checkpoint import CheckpointError
from kinera.training.hybrid_checkpoint import (
    _architecture,
    _parameters,
    _validate_training_state,
    load_autoregressive_checkpoint,
    save_autoregressive_checkpoint,
)
from kinera.training.trainer import train_examples, train_multitask_examples


TrainingMode = Literal["char", "multitask"]


def _context(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    *,
    mode: TrainingMode,
    learning_rate: float,
    max_examples: int | None,
    syllable_unknown_id: int,
    loss_function: MultiTaskLoss | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], MultiTaskLoss | None]:
    if not isinstance(model, AutoregressiveModel) or not isinstance(model.backend, CPUBackend):
        raise CheckpointError("R2.4 suporta apenas AutoregressiveModel com CPUBackend")
    if model._closed:
        raise CheckpointError("modelo fechado não pode ser treinado")
    if not isinstance(character_vocabulary, CharacterVocabulary):
        raise TypeError("character_vocabulary inválido")
    if not isinstance(syllable_vocabulary, SyllableVocabulary):
        raise TypeError("syllable_vocabulary inválido")
    if model.representation.char_embedding.vocab_size != len(character_vocabulary):
        raise CheckpointError("tamanho do vocabulário de caracteres incompatível")
    if model.representation.syllable_embedding.vocab_size != len(syllable_vocabulary):
        raise CheckpointError("tamanho do vocabulário silábico incompatível")
    # Existing example_to_inputs in both trainers uses feature_dim=10.
    if model.representation.feature_dim != 10:
        raise CheckpointError("treinamento canônico requer feature_dim=10")
    if mode not in ("char", "multitask"):
        raise CheckpointError("modo de treino não suportado")
    if type(learning_rate) not in (float, int):
        raise CheckpointError("learning_rate inválido")
    rate = float(learning_rate)
    import math
    if not math.isfinite(rate) or rate <= 0:
        raise CheckpointError("learning_rate deve ser positivo e finito")
    if max_examples is not None and (type(max_examples) is not int or max_examples <= 0):
        raise CheckpointError("max_examples deve ser positivo ou None")
    if type(syllable_unknown_id) is not int or not 0 <= syllable_unknown_id < len(syllable_vocabulary):
        raise CheckpointError("syllable_unknown_id incompatível")
    if not isinstance(examples, list) or not examples:
        raise CheckpointError("examples deve ser lista não vazia")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected or any(not isinstance(item, dict) for item in selected):
        raise CheckpointError("exemplos selecionados inválidos")

    if mode == "char":
        if loss_function is not None:
            raise CheckpointError("loss_function não é utilizada pelo treino char")
        loss_weights = None
        loss = None
    else:
        loss = MultiTaskLoss(backend=model.backend) if loss_function is None else loss_function
        if not isinstance(loss, MultiTaskLoss) or not isinstance(loss.backend, CPUBackend):
            raise CheckpointError("loss_function deve ser MultiTaskLoss CPU")
        loss_weights = dict(loss.weights)

    try:
        data_digest = canonical_sha256(selected)
    except (ValueError, TypeError, OverflowError) as error:
        raise CheckpointError(f"exemplos não têm identidade canônica válida: {error}") from error

    return ({
        "schema_version": 1,
        "mode": mode,
        "selected_examples": len(selected),
        "examples_sha256": data_digest,
        "configuration": {
            "learning_rate": rate,
            "max_examples": max_examples,
            "syllable_unknown_id": syllable_unknown_id,
            "loss_weights": loss_weights,
        },
    }, selected, loss)


def _parameters_digest(model: AutoregressiveModel) -> str:
    """Bind a resume cursor to the exact eight live CPU parameter arrays."""
    return canonical_sha256(_parameters(model, _architecture(model)))


def _verify_state(
    state: dict[str, Any],
    context: dict[str, Any],
    model: AutoregressiveModel,
) -> dict[str, Any]:
    _validate_training_state(state)
    for key in ("schema_version", "mode", "selected_examples", "examples_sha256", "configuration"):
        if state[key] != context[key]:
            raise CheckpointError(f"retomada incompatível: {key}")
    if state["model_parameters_sha256"] != _parameters_digest(model):
        raise CheckpointError("retomada incompatível: parâmetros não correspondem ao estado")
    return state


def train_hybrid_epochs(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    *,
    mode: TrainingMode,
    epochs: int,
    learning_rate: float,
    max_examples: int | None = None,
    syllable_unknown_id: int = 0,
    loss_function: MultiTaskLoss | None = None,
    previous_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train *additional* whole epochs and return aggregate report and state.

    No mid-epoch resume, shuffle, callbacks, timing sessions, or RNG state:
    these two canonical trainers consume examples in fixed order and perform
    plain per-example SGD. Do not substitute this wrapper for runner observers.
    """
    if type(epochs) is not int or epochs <= 0:
        raise CheckpointError("epochs adicionais devem ser inteiros positivos")
    context, _, loss = _context(
        model, examples, character_vocabulary, syllable_vocabulary,
        mode=mode, learning_rate=learning_rate, max_examples=max_examples,
        syllable_unknown_id=syllable_unknown_id, loss_function=loss_function,
    )
    if previous_state is None:
        state: dict[str, Any] = {
            **context,
            "completed_epochs": 0,
            "global_step": 0,
            "model_parameters_sha256": _parameters_digest(model),
            "history": [],
        }
    else:
        state = deepcopy(_verify_state(previous_state, context, model))

    for _ in range(epochs):
        if mode == "char":
            report = train_examples(
                model, examples, character_vocabulary,
                epochs=1, learning_rate=learning_rate,
                syllable_unknown_id=syllable_unknown_id, max_examples=max_examples,
            )
        else:
            report = train_multitask_examples(
                model, examples, character_vocabulary,
                epochs=1, learning_rate=learning_rate,
                syllable_unknown_id=syllable_unknown_id, max_examples=max_examples,
                loss_function=loss,
            )
        state["history"].extend(report["history"])
        state["completed_epochs"] += 1
        state["global_step"] += context["selected_examples"]
        state["model_parameters_sha256"] = _parameters_digest(model)
    _verify_state(state, context, model)
    result = {
        "epochs": state["completed_epochs"],
        "examples": context["selected_examples"],
        "history": deepcopy(state["history"]),
        "global_step": state["global_step"],
    }
    return result, deepcopy(state)


def save_hybrid_training_checkpoint(
    path: str | Path,
    model: AutoregressiveModel,
    state: dict[str, Any],
    *,
    examples: list[dict[str, Any]],
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    mode: TrainingMode,
    learning_rate: float,
    max_examples: int | None = None,
    syllable_unknown_id: int = 0,
    loss_function: MultiTaskLoss | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically persist a validated snapshot after a full CPU epoch."""
    context, _, _ = _context(
        model, examples, character_vocabulary, syllable_vocabulary,
        mode=mode, learning_rate=learning_rate, max_examples=max_examples,
        syllable_unknown_id=syllable_unknown_id, loss_function=loss_function,
    )
    validated = _verify_state(state, context, model)
    if validated["completed_epochs"] < 1:
        raise CheckpointError("não é possível salvar estado antes da primeira época")
    return save_autoregressive_checkpoint(
        path, model,
        character_vocabulary=character_vocabulary,
        syllable_vocabulary=syllable_vocabulary,
        metadata=metadata,
        training_state=deepcopy(validated),
    )


def load_hybrid_training_checkpoint(
    path: str | Path,
    *,
    examples: list[dict[str, Any]],
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
    mode: TrainingMode,
    learning_rate: float,
    max_examples: int | None = None,
    syllable_unknown_id: int = 0,
    loss_function: MultiTaskLoss | None = None,
) -> tuple[AutoregressiveModel, dict[str, Any], dict[str, Any]]:
    """Fail closed on data/config/vocab drift before allowing further updates."""
    model, info = load_autoregressive_checkpoint(
        path,
        expected_character_vocabulary=character_vocabulary,
        expected_syllable_vocabulary=syllable_vocabulary,
    )
    state = info["training"]
    if state is None:
        raise CheckpointError("checkpoint contém apenas inferência; estado de treino ausente")
    context, _, _ = _context(
        model, examples, character_vocabulary, syllable_vocabulary,
        mode=mode, learning_rate=learning_rate, max_examples=max_examples,
        syllable_unknown_id=syllable_unknown_id, loss_function=loss_function,
    )
    _verify_state(state, context, model)
    return model, deepcopy(state), info["metadata"]


__all__ = [
    "train_hybrid_epochs",
    "save_hybrid_training_checkpoint",
    "load_hybrid_training_checkpoint",
]
