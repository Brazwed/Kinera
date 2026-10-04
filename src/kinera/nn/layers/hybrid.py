"""Composição da representação híbrida por caractere."""

from __future__ import annotations

import math
from typing import Sequence, cast

from kinera.backends.base import (
    BackendArray,
    BackendBuffer,
    BackendUnavailableError,
    release_backend_buffers,
)
from kinera.nn.layers.embedding import Embedding


class HybridRepresentation:
    """Concatena embedding de caractere, features e embedding de sílaba."""

    def __init__(
        self,
        char_embedding: Embedding,
        syllable_embedding: Embedding,
        *,
        feature_dim: int,
    ) -> None:
        if not isinstance(char_embedding, Embedding):
            raise TypeError("char_embedding deve ser Embedding")
        if not isinstance(syllable_embedding, Embedding):
            raise TypeError("syllable_embedding deve ser Embedding")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim deve ser inteiro")
        if feature_dim <= 0:
            raise ValueError("feature_dim deve ser positivo")
        self.char_embedding = char_embedding
        self.syllable_embedding = syllable_embedding
        self.feature_dim = feature_dim
        self.last_output: list[list[float]] | BackendBuffer | None = None

    @property
    def output_dim(self) -> int:
        return (
            self.char_embedding.embedding_dim
            + self.feature_dim
            + self.syllable_embedding.embedding_dim
        )

    def release_output(self) -> None:
        """Release the latest owned hybrid matrix when its consumer is done."""
        release_backend_buffers(self.last_output)
        self.last_output = None

    def forward(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> list[list[float]] | BackendBuffer:
        self._validate_parallel_inputs(char_ids, syllable_ids, feature_vectors)
        self.release_output()
        feature_buffer: BackendArray | None = None
        try:
            char_vectors = self.char_embedding.forward(char_ids)
            syllable_vectors = self.syllable_embedding.forward(syllable_ids)
            if not char_ids:
                if isinstance(char_vectors, BackendBuffer) or isinstance(
                    syllable_vectors, BackendBuffer
                ):
                    raise BackendUnavailableError(
                        "empty hybrid inputs are not supported by device buffers"
                    )
                output = []
            else:
                feature_buffer = self.char_embedding.backend.buffer_from_host(
                    feature_vectors,
                    shape=(len(feature_vectors), self.feature_dim),
                    dtype="float64",
                )
                hybrid_forward = getattr(
                    self.char_embedding.backend, "hybrid_representation_forward", None
                )
                if callable(hybrid_forward):
                    output = hybrid_forward(
                        char_vectors,
                        cast(BackendArray, feature_buffer),
                        syllable_vectors,
                    )
                elif any(
                    isinstance(value, BackendBuffer)
                    for value in (char_vectors, feature_buffer, syllable_vectors)
                ):
                    raise BackendUnavailableError(
                        "backend must provide hybrid_representation_forward for device buffers"
                    )
                else:
                    char_values = cast(list[list[float]], char_vectors)
                    feature_values = cast(list[list[float]], feature_buffer)
                    syllable_values = cast(list[list[float]], syllable_vectors)
                    output = [
                        char_vector + feature_vector + syllable_vector
                        for char_vector, feature_vector, syllable_vector in zip(
                            char_values, feature_values, syllable_values
                        )
                    ]
        except Exception:
            release_backend_buffers(feature_buffer)
            self.char_embedding.discard_forward_state()
            self.syllable_embedding.discard_forward_state()
            raise

        release_backend_buffers(feature_buffer)
        self.char_embedding.release_output_buffer()
        self.syllable_embedding.release_output_buffer()
        if isinstance(output, BackendBuffer):
            self.last_output = output
            return output

        output_values = cast(list[list[float]], output)
        self.last_output = [vector.copy() for vector in output_values]
        return [vector.copy() for vector in output_values]

    def backward(self, grad_output: list[list[float]]) -> list[list[float]]:
        """Propaga gradientes às embeddings e devolve gradientes das features."""
        if self.last_output is None:
            raise RuntimeError("backward exige um forward anterior")
        if isinstance(self.last_output, BackendBuffer):
            raise BackendUnavailableError(
                "CUDA operation HybridRepresentation.backward is not available in Phase 11A"
            )
        if not isinstance(grad_output, list):
            raise TypeError("grad_output deve ser uma lista")
        if len(grad_output) != len(self.last_output):
            raise ValueError("grad_output deve ter o mesmo número de posições")

        char_dim = self.char_embedding.embedding_dim
        syllable_dim = self.syllable_embedding.embedding_dim
        char_gradients: list[list[float]] = []
        syllable_gradients: list[list[float]] = []
        feature_gradients: list[list[float]] = []
        for position, vector in enumerate(grad_output):
            if not isinstance(vector, list) or len(vector) != self.output_dim:
                raise ValueError(
                    f"grad_output[{position}] deve ter dimensão {self.output_dim}"
                )
            self._validate_vector(vector, f"grad_output[{position}]")
            char_gradients.append(vector[:char_dim])
            feature_gradients.append(vector[char_dim : char_dim + self.feature_dim])
            syllable_gradients.append(vector[char_dim + self.feature_dim :])

        self.char_embedding.backward(char_gradients)
        self.syllable_embedding.backward(syllable_gradients)
        return feature_gradients

    def step(self, learning_rate: float) -> None:
        self.char_embedding.step(learning_rate)
        self.syllable_embedding.step(learning_rate)

    def _validate_parallel_inputs(
        self,
        char_ids: list[int],
        syllable_ids: list[int],
        feature_vectors: list[list[float]],
    ) -> None:
        if not isinstance(char_ids, list) or not isinstance(syllable_ids, list):
            raise TypeError("IDs devem ser listas")
        if not isinstance(feature_vectors, list):
            raise TypeError("feature_vectors deve ser uma lista")
        if not (len(char_ids) == len(syllable_ids) == len(feature_vectors)):
            raise ValueError("entradas devem ter o mesmo comprimento")
        for position, vector in enumerate(feature_vectors):
            if not isinstance(vector, list) or len(vector) != self.feature_dim:
                raise ValueError(
                    f"feature_vectors[{position}] deve ter dimensão {self.feature_dim}"
                )
            self._validate_vector(vector, f"feature_vectors[{position}]")

    @staticmethod
    def _validate_vector(vector: Sequence[object], name: str) -> None:
        for index, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name}[{index}] deve ser numérico")
            if not math.isfinite(value):
                raise ValueError(f"{name}[{index}] deve ser finito")
