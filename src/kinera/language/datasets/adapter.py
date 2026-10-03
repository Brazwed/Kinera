"""Adapters from persisted examples to aligned model inputs."""

from __future__ import annotations

import math
from typing import Any

from kinera.language.orthography.features import character_features, features_to_vector
from kinera.language.vocabulary import CharacterVocabulary


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
