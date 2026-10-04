"""Hybrid autoregressive model independent of CPU/CUDA implementations."""

from __future__ import annotations

import math
from typing import Mapping, cast

from kinera.backends import Backend, resolve_backend
from kinera.backends.base import (
    BackendArray,
    BackendBuffer,
    BackendError,
    BackendUnavailableError,
    release_backend_buffers,
)
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
        self.last_logits: list[float] | BackendBuffer | None = None
        self.last_pooled: list[float] | BackendBuffer | None = None
        self._last_head_logits_backend: dict[str, list[float] | BackendBuffer] = {}
        self.last_context_size: int | None = None

    def forward(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> list[float]:
        """Calcula logits para o próximo caractere na fronteira pública do host."""
        return self._export_vector(
            self._forward_backend(char_ids, syllable_ids, feature_vectors)
        )

    def forward_heads(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> dict[str, list[float]]:
        """Calcula e exporta logits das três cabeças para o host."""
        outputs = self._forward_heads_backend(
            char_ids, syllable_ids, feature_vectors
        )
        return {
            name: self._export_vector(values) for name, values in outputs.items()
        }

    def _forward_backend(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> list[float] | BackendBuffer:
        """Retorna logits nativos; buffers emprestados até novo forward ou release_forward_state."""
        self._validate_context(char_ids, syllable_ids, feature_vectors)
        self.release_forward_state()
        pooled: list[float] | BackendBuffer | None = None
        logits: list[float] | BackendBuffer | None = None
        try:
            pooled = self._pool_representation_backend(
                char_ids, syllable_ids, feature_vectors
            )
            logits = self.output_layer.forward(pooled)
        except Exception:
            release_backend_buffers(pooled, logits)
            self.release_forward_state()
            raise

        self.last_pooled = self._retain_vector(pooled)
        self.last_logits = self._retain_vector(logits)
        self._last_head_logits_backend = {"char": logits}
        self.last_context_size = len(char_ids)
        return logits

    def _forward_heads_backend(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> dict[str, list[float] | BackendBuffer]:
        """Retorna logits de todas as heads; buffers emprestados até novo forward ou release_forward_state."""
        self._validate_context(char_ids, syllable_ids, feature_vectors)
        self.release_forward_state()
        pooled: list[float] | BackendBuffer | None = None
        outputs: dict[str, list[float] | BackendBuffer] = {}
        try:
            pooled = self._pool_representation_backend(
                char_ids, syllable_ids, feature_vectors
            )
            outputs = {
                "char": self.heads.char.forward(pooled),
                "syllable": self.heads.syllable.forward(pooled),
                "boundary": self.heads.boundary.forward(pooled),
            }
        except Exception:
            release_backend_buffers(pooled, *outputs.values())
            self.release_forward_state()
            raise

        self.last_pooled = self._retain_vector(pooled)
        self.last_logits = self._retain_vector(outputs["char"])
        self._last_head_logits_backend = outputs.copy()
        self.last_context_size = len(char_ids)
        return outputs

    def _pool_representation_backend(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> list[float] | BackendBuffer:
        vectors = self.representation.forward(
            char_ids, syllable_ids, feature_vectors
        )
        pooled: BackendArray | None = None
        try:
            mean_pool = getattr(self.backend, "mean_pool_forward", None)
            if callable(mean_pool):
                pooled = cast(BackendArray, mean_pool(vectors))
            elif isinstance(vectors, BackendBuffer):
                raise BackendUnavailableError(
                    "backend must provide mean_pool_forward for device buffers"
                )
            else:
                vector_values = cast(list[list[float]], vectors)
                context_size = len(vector_values)
                pooled = [
                    sum(vector[index] for vector in vector_values) / context_size
                    for index in range(self.representation.output_dim)
                ]
            if isinstance(pooled, BackendBuffer):
                if pooled.closed or pooled.dtype != "float64":
                    raise BackendError("mean pooling returned an invalid device buffer")
                if pooled.shape != (self.representation.output_dim,):
                    raise BackendError("mean pooling returned an invalid buffer shape")
                return pooled
            if not isinstance(pooled, list):
                raise TypeError("mean pooling must return a list or BackendBuffer")
            if len(pooled) != self.representation.output_dim:
                raise ValueError("mean pooling returned an invalid vector size")
            return cast(list[float], pooled)
        except Exception:
            release_backend_buffers(pooled)
            raise
        finally:
            if isinstance(vectors, BackendBuffer):
                self.representation.release_output()

    def release_forward_state(self) -> None:
        """Fechar caches/ativações retidos; os parâmetros do modelo continuam válidos."""
        embeddings = (
            self.representation.char_embedding,
            self.representation.syllable_embedding,
        )
        heads = (self.heads.char, self.heads.syllable, self.heads.boundary)
        buffers: list[object] = [
            self.last_pooled,
            self.last_logits,
            self.representation.last_output,
            *self._last_head_logits_backend.values(),
        ]
        for embedding in embeddings:
            buffers.extend(
                (embedding.last_ids, embedding.output, embedding.grad_weights)
            )
        for head in heads:
            buffers.extend(
                (head.last_input, head.output, head.grad_weights, head.grad_bias)
            )
        release_backend_buffers(*buffers)

        self.representation.discard_forward_state()
        for embedding in embeddings:
            embedding.discard_forward_state()
        for head in heads:
            head.last_input = None
            head.output = None
            head.release_backward_state()

        self.last_logits = None
        self.last_pooled = None
        self._last_head_logits_backend = {}
        self.last_context_size = None

    @staticmethod
    def _retain_vector(
        values: list[float] | BackendBuffer,
    ) -> list[float] | BackendBuffer:
        if isinstance(values, BackendBuffer):
            return values
        return values.copy()

    def _export_vector(self, values: list[float] | BackendBuffer) -> list[float]:
        host_values = (
            self.backend.buffer_to_host(values)
            if isinstance(values, BackendBuffer)
            else values
        )
        return cast(list[float], host_values).copy()

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
        temporary_buffers: list[object] = []
        try:
            outputs = self._forward_heads_backend(
                char_ids, syllable_ids, feature_vectors
            )
            report = loss_function.forward(outputs, targets)
            logit_gradients = loss_function.backward(outputs, targets)
            temporary_buffers.extend(logit_gradients.values())
            input_gradients = self.heads.backward(logit_gradients)
            temporary_buffers.extend(input_gradients.values())
            context_size = self.last_context_size
            if context_size is None:
                raise RuntimeError("multitask backward lost the forward context size")

            head_names = ("char", "syllable", "boundary")
            resident_flags = [
                isinstance(input_gradients[name], BackendBuffer)
                for name in head_names
            ]
            if any(resident_flags):
                if not all(resident_flags):
                    raise BackendError(
                        "multitask head gradients must all use the same backend storage"
                    )
                aggregate = getattr(self.backend, "sum_head_input_gradients", None)
                if not callable(aggregate):
                    raise BackendUnavailableError(
                        "backend must provide sum_head_input_gradients for resident gradients"
                    )
                pooled_gradient = aggregate(
                    input_gradients["char"],
                    input_gradients["syllable"],
                    input_gradients["boundary"],
                )
                if not isinstance(pooled_gradient, BackendBuffer):
                    release_backend_buffers(pooled_gradient)
                    raise BackendError(
                        "sum_head_input_gradients must return a BackendBuffer"
                    )
                temporary_buffers.append(pooled_gradient)
                if pooled_gradient.closed or pooled_gradient.dtype != "float64":
                    raise BackendError("head aggregation returned an invalid buffer")
                if pooled_gradient.shape != (self.representation.output_dim,):
                    raise BackendError("head aggregation returned an invalid shape")

                pool_backward = getattr(self.backend, "mean_pool_backward", None)
                if not callable(pool_backward):
                    raise BackendUnavailableError(
                        "backend must provide mean_pool_backward for resident gradients"
                    )
                representation_gradient = pool_backward(
                    pooled_gradient, context_size=context_size
                )
                if not isinstance(representation_gradient, BackendBuffer):
                    release_backend_buffers(representation_gradient)
                    raise BackendError("mean_pool_backward must return a BackendBuffer")
                temporary_buffers.append(representation_gradient)
                if representation_gradient.closed or representation_gradient.dtype != "float64":
                    raise BackendError("mean_pool_backward returned an invalid buffer")
                if representation_gradient.shape != (
                    context_size,
                    self.representation.output_dim,
                ):
                    raise BackendError("mean_pool_backward returned an invalid shape")
                feature_gradients = self.representation.backward(
                    representation_gradient
                )
                temporary_buffers.append(feature_gradients)
            else:
                char_gradient = cast(list[float], input_gradients["char"])
                syllable_gradient = cast(list[float], input_gradients["syllable"])
                boundary_gradient = cast(list[float], input_gradients["boundary"])
                pooled_gradient = [
                    sum(
                        (
                            char_gradient[index],
                            syllable_gradient[index],
                            boundary_gradient[index],
                        )
                    )
                    / context_size
                    for index in range(self.representation.output_dim)
                ]
                self.representation.backward(
                    [pooled_gradient.copy() for _ in range(context_size)]
                )

            self.heads.step(learning_rate)
            self.representation.step(learning_rate)
            return report
        finally:
            release_backend_buffers(*temporary_buffers)
            if self.backend.name == "cuda":
                self.release_forward_state()

    def predict(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> int:
        """Retorna o ID previsto sem exportar os logits ao host."""
        logits = self._forward_backend(char_ids, syllable_ids, feature_vectors)
        argmax = getattr(self.backend, "argmax", None)
        if callable(argmax):
            return cast(int, argmax(logits))
        if isinstance(logits, BackendBuffer):
            raise BackendUnavailableError(
                "backend must provide argmax for device-resident logits"
            )
        logits_values = cast(list[float], logits)
        return max(range(len(logits_values)), key=logits_values.__getitem__)

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
