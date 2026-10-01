"""Modelo autoregressivo mínimo sobre a representação híbrida."""

from __future__ import annotations

import math
from typing import Any

from nn.embedding import Embedding
from nn.hybrid import HybridRepresentation
from nn.linear import Linear
from nn.probability import cross_entropy, cross_entropy_gradient
from language.features import character_features, features_to_vector
from language.vocabulary import CharacterVocabulary


class AutoregressiveModel:
    """Prevê o próximo token a partir da média de um contexto híbrido."""

    def __init__(
        self,
        char_vocab_size: int,
        syllable_vocab_size: int,
        *,
        char_embedding_dim: int = 8,
        syllable_embedding_dim: int = 8,
        feature_dim: int = 10,
        seed: int | None = None,
    ) -> None:
        self.representation = HybridRepresentation(
            Embedding(char_vocab_size, char_embedding_dim, seed=seed),
            Embedding(syllable_vocab_size, syllable_embedding_dim, seed=None if seed is None else seed + 1),
            feature_dim=feature_dim,
        )
        self.output_layer = Linear(
            self.representation.output_dim,
            char_vocab_size,
            seed=None if seed is None else seed + 2,
        )
        self.last_logits: list[float] | None = None
        self.last_pooled: list[float] | None = None
        self.last_context_size: int | None = None

    def forward(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> list[float]:
        """Calcula logits para o próximo caractere."""
        self._validate_context(char_ids, syllable_ids, feature_vectors)
        vectors = self.representation.forward(
            char_ids,
            syllable_ids,
            feature_vectors,
        )
        context_size = len(vectors)
        pooled = [
            sum(vector[index] for vector in vectors) / context_size
            for index in range(self.representation.output_dim)
        ]
        self.last_pooled = pooled.copy()
        self.last_context_size = context_size
        self.last_logits = self.output_layer.forward(pooled)
        return self.last_logits.copy()

    def loss(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        *,
        target_id: int,
    ) -> float:
        """Calcula a loss sem alterar os parâmetros."""
        logits = self.forward(char_ids, syllable_ids, feature_vectors)
        return cross_entropy(logits, target_id)

    def train_step(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        *,
        target_id: int,
        learning_rate: float,
    ) -> float:
        """Executa forward, backward e atualização para um exemplo."""
        logits = self.forward(char_ids, syllable_ids, feature_vectors)
        loss = cross_entropy(logits, target_id)
        gradient_logits = cross_entropy_gradient(logits, target_id)
        gradient_pooled = self.output_layer.backward(gradient_logits)
        assert self.last_context_size is not None
        gradient_vector = [
            value / self.last_context_size for value in gradient_pooled
        ]
        self.representation.backward([
            gradient_vector.copy() for _ in range(self.last_context_size)
        ])
        self.output_layer.step(learning_rate)
        self.representation.step(learning_rate)
        return loss

    def predict(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> int:
        """Retorna o ID do token com maior logit."""
        logits = self.forward(char_ids, syllable_ids, feature_vectors)
        return max(range(len(logits)), key=logits.__getitem__)

    def _validate_context(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> None:
        if not char_ids:
            raise ValueError("o contexto não pode ser vazio")
        if not isinstance(char_ids, list) or not isinstance(syllable_ids, list):
            raise TypeError("IDs devem ser listas")
        if not isinstance(feature_vectors, list):
            raise TypeError("feature_vectors deve ser uma lista")
        if not (len(char_ids) == len(syllable_ids) == len(feature_vectors)):
            raise ValueError("entradas devem ter o mesmo comprimento")
        for vector in feature_vectors:
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError("features devem ser numéricas")
                if not math.isfinite(value):
                    raise ValueError("features devem ser finitas")


def example_to_inputs(
    example: dict[str, Any],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int,
    feature_dim: int = 10,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Converte um exemplo JSONL em entradas alinhadas do modelo.

    Exemplos antigos não carregam alinhamento silábico por posição. Nesses
    casos, a sílaba recebe o ID desconhecido, enquanto as features do
    caractere continuam sendo usadas. Artefatos futuros podem substituir
    esses IDs sem mudar a interface do modelo.
    """
    if not isinstance(example, dict):
        raise TypeError("example deve ser um dicionário")
    context_ids = example.get("context_ids")
    if not isinstance(context_ids, list) or not context_ids:
        raise ValueError("example deve conter context_ids não vazio")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
        raise TypeError("feature_dim deve ser inteiro")
    if feature_dim <= 0:
        raise ValueError("feature_dim deve ser positivo")

    char_ids: list[int] = []
    syllable_ids: list[int] = []
    feature_vectors: list[list[float]] = []
    persisted_syllable_ids = example.get("context_syllable_ids")
    persisted_features = example.get("context_feature_vectors")
    stored_syllable_ids: list[object] = []
    stored_features: list[object] = []
    has_persisted_context = (
        persisted_syllable_ids is not None or persisted_features is not None
    )
    if has_persisted_context:
        if not isinstance(persisted_syllable_ids, list) or not isinstance(
            persisted_features, list
        ):
            raise ValueError(
                "context_syllable_ids e context_feature_vectors devem ser listas"
            )
        if len(persisted_syllable_ids) != len(context_ids) or len(
            persisted_features
        ) != len(context_ids):
            raise ValueError("contexto persistido tem comprimento inválido")
        stored_syllable_ids = persisted_syllable_ids
        stored_features = persisted_features
    for position, char_id in enumerate(context_ids):
        if isinstance(char_id, bool) or not isinstance(char_id, int):
            raise TypeError(f"context_ids[{position}] deve ser inteiro")
        if char_id not in vocabulary.id_to_token:
            raise ValueError(f"context_ids[{position}] fora do vocabulário")
        token = vocabulary.id_to_token[char_id]
        if has_persisted_context:
            syllable_id = stored_syllable_ids[position]
            vector = stored_features[position]
            if isinstance(syllable_id, bool) or not isinstance(syllable_id, int):
                raise TypeError("IDs silábicos persistidos devem ser inteiros")
            if not isinstance(vector, list) or len(vector) != feature_dim:
                raise ValueError("features persistidas têm dimensão inválida")
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError("features persistidas devem ser numéricas")
                if not math.isfinite(value):
                    raise ValueError("features persistidas devem ser finitas")
            syllable_ids.append(syllable_id)
            feature_vectors.append([float(value) for value in vector])
            char_ids.append(char_id)
            continue
        if len(token) == 1:
            vector = features_to_vector(character_features(token))
        else:
            vector = [0.0] * feature_dim
        if len(vector) != feature_dim:
            raise ValueError("feature_dim incompatível com as features ortográficas")
        char_ids.append(char_id)
        syllable_ids.append(syllable_unknown_id)
        feature_vectors.append(vector)
    return char_ids, syllable_ids, feature_vectors


def train_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    epochs: int,
    learning_rate: float,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Treina exemplos em ordem determinística e retorna métricas por época."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs deve ser inteiro positivo")
    if not examples:
        raise ValueError("examples não pode ser vazio")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected:
        raise ValueError("max_examples deve selecionar ao menos um exemplo")

    history: list[dict[str, float]] = []
    for _ in range(epochs):
        total_loss = 0.0
        correct = 0
        for example in selected:
            inputs = example_to_inputs(
                example,
                vocabulary,
                syllable_unknown_id=syllable_unknown_id,
            )
            target_id = example.get("target_char_id")
            if isinstance(target_id, bool) or not isinstance(target_id, int):
                raise ValueError("target_char_id deve ser inteiro")
            prediction = model.predict(*inputs)
            total_loss += model.train_step(
                *inputs,
                target_id=target_id,
                learning_rate=learning_rate,
            )
            correct += prediction == target_id
        history.append(
            {
                "loss": total_loss / len(selected),
                "accuracy": correct / len(selected),
            }
        )
    return {"epochs": epochs, "examples": len(selected), "history": history}


def evaluate_examples(
    model: AutoregressiveModel,
    examples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
    *,
    syllable_unknown_id: int = 0,
    max_examples: int | None = None,
) -> dict[str, float | int]:
    """Mede loss e acurácia sem executar backward ou step."""
    if not examples:
        raise ValueError("examples não pode ser vazio")
    selected = examples if max_examples is None else examples[:max_examples]
    if not selected:
        raise ValueError("max_examples deve selecionar ao menos um exemplo")

    total_loss = 0.0
    correct = 0
    for example in selected:
        inputs = example_to_inputs(
            example,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        target_id = example.get("target_char_id")
        if isinstance(target_id, bool) or not isinstance(target_id, int):
            raise ValueError("target_char_id deve ser inteiro")
        logits = model.forward(*inputs)
        total_loss += cross_entropy(logits, target_id)
        correct += max(range(len(logits)), key=logits.__getitem__) == target_id
    return {
        "examples": len(selected),
        "loss": total_loss / len(selected),
        "accuracy": correct / len(selected),
    }


def generate_text(
    model: AutoregressiveModel,
    vocabulary: CharacterVocabulary,
    *,
    context_size: int,
    max_tokens: int,
    syllable_unknown_id: int = 0,
) -> str:
    """Gera texto de forma gulosa até ``<END>`` ou o limite definido."""
    if context_size <= 0 or max_tokens < 0:
        raise ValueError("context_size deve ser positivo e max_tokens não negativo")
    context = [vocabulary.start_id] * context_size
    generated: list[int] = []
    for _ in range(max_tokens):
        example = {"context_ids": context}
        inputs = example_to_inputs(
            example,
            vocabulary,
            syllable_unknown_id=syllable_unknown_id,
        )
        next_id = model.predict(*inputs)
        if next_id == vocabulary.end_id:
            break
        generated.append(next_id)
        context = [*context[1:], next_id]
    return vocabulary.decode(generated, skip_special=True)
