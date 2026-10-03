"""Hybrid autoregressive model independent of CPU/CUDA implementations."""

from __future__ import annotations

import math
from typing import Mapping

from kinera.backends import Backend, resolve_backend
from kinera.nn.heads.multitask import MultiTaskHeads
from kinera.nn.layers.embedding import Embedding
from kinera.nn.layers.hybrid import HybridRepresentation
from kinera.nn.losses.multitask import MultiTaskLoss
from kinera.nn.losses.cross_entropy import cross_entropy, cross_entropy_gradient


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
        backend: Backend | None = None,
    ) -> None:
        self.backend = resolve_backend(backend)
        self.representation = HybridRepresentation(
            Embedding(
                char_vocab_size,
                char_embedding_dim,
                seed=seed,
                backend=self.backend,
            ),
            Embedding(
                syllable_vocab_size,
                syllable_embedding_dim,
                seed=None if seed is None else seed + 1,
                backend=self.backend,
            ),
            feature_dim=feature_dim,
        )
        self.heads = MultiTaskHeads(
            self.representation.output_dim,
            char_vocab_size,
            syllable_vocab_size,
            seed=None if seed is None else seed + 2,
            backend=self.backend,
        )
        self.output_layer = self.heads.char
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
        pooled = self._forward_representation(
            char_ids, syllable_ids, feature_vectors
        )
        self.last_logits = self.output_layer.forward(pooled)
        return self.last_logits.copy()

    def forward_heads(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> dict[str, list[float]]:
        """Calcula logits das cabeças de caractere, sílaba e fronteira."""
        self._validate_context(char_ids, syllable_ids, feature_vectors)
        pooled = self._forward_representation(
            char_ids, syllable_ids, feature_vectors
        )
        outputs = self.heads.forward(pooled)
        self.last_logits = outputs["char"].copy()
        return {name: values.copy() for name, values in outputs.items()}

    def _forward_representation(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> list[float]:
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
        return pooled

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
        return cross_entropy(logits, target_id, backend=self.backend)

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
        loss = cross_entropy(logits, target_id, backend=self.backend)
        gradient_logits = cross_entropy_gradient(
            logits, target_id, backend=self.backend
        )
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

    def multitask_train_step(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
        *,
        targets: Mapping[str, object],
        learning_rate: float,
        loss_function: MultiTaskLoss | None = None,
    ) -> dict[str, float]:
        """Executa forward/backward/step para as três cabeças."""
        loss_function = (
            MultiTaskLoss(backend=self.backend)
            if loss_function is None
            else loss_function
        )
        outputs = self.forward_heads(char_ids, syllable_ids, feature_vectors)
        report = loss_function.forward(outputs, targets)
        gradients = loss_function.backward(outputs, targets)
        input_gradients = self.heads.backward(gradients)
        assert self.last_context_size is not None
        pooled_gradient = [
            sum(input_gradients[name][index] for name in input_gradients)
            / self.last_context_size
            for index in range(self.representation.output_dim)
        ]
        self.representation.backward([
            pooled_gradient.copy() for _ in range(self.last_context_size)
        ])
        self.heads.step(learning_rate)
        self.representation.step(learning_rate)
        return report

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

__all__ = ["AutoregressiveModel"]
