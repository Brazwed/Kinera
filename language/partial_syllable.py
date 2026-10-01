"""Codificação causal de sílabas em construção."""

from __future__ import annotations

import hashlib
import copy
from typing import Any

from language.vocabulary import CharacterVocabulary, SyllableVocabulary


class CausalPartialSyllableEncoder:
    """Converte prefixos observados em IDs sem consultar caracteres futuros.

    IDs de sílabas já concluídas são preservados. Enquanto a sílaba atual está
    incompleta, somente os caracteres observados desde a última conclusão são
    acumulados e enviados para buckets hash determinísticos. O tamanho da
    tabela é exatamente o da tabela silábica existente para não alterar a
    contagem de parâmetros do modelo.
    """

    def __init__(
        self,
        character_vocabulary: CharacterVocabulary,
        syllable_vocabulary: SyllableVocabulary,
    ) -> None:
        if not isinstance(character_vocabulary, CharacterVocabulary):
            raise TypeError("character_vocabulary inválido")
        if not isinstance(syllable_vocabulary, SyllableVocabulary):
            raise TypeError("syllable_vocabulary inválido")
        self.character_vocabulary = character_vocabulary
        self.syllable_vocabulary = syllable_vocabulary

    def encode_context(
        self,
        char_ids: list[int],
        completed_syllable_ids: list[int],
    ) -> list[int]:
        if not isinstance(char_ids, list) or not isinstance(completed_syllable_ids, list):
            raise TypeError("char_ids e completed_syllable_ids devem ser listas")
        if len(char_ids) != len(completed_syllable_ids):
            raise ValueError("entradas devem ter o mesmo comprimento")
        prefix = ""
        encoded: list[int] = []
        for position, (char_id, syllable_id) in enumerate(
            zip(char_ids, completed_syllable_ids)
        ):
            if isinstance(char_id, bool) or not isinstance(char_id, int):
                raise TypeError(f"char_ids[{position}] deve ser inteiro")
            if char_id not in self.character_vocabulary.id_to_token:
                raise ValueError(f"char_ids[{position}] fora do vocabulário")
            if isinstance(syllable_id, bool) or not isinstance(syllable_id, int):
                raise TypeError(f"completed_syllable_ids[{position}] deve ser inteiro")
            if syllable_id != self.syllable_vocabulary.unk_id:
                if not 0 <= syllable_id < len(self.syllable_vocabulary):
                    raise ValueError(f"completed_syllable_ids[{position}] fora do vocabulário")
                prefix = ""
                encoded.append(syllable_id)
                continue
            token = self.character_vocabulary.id_to_token[char_id]
            if token in {"<START>", "<END>", "<UNK>", "-", "|"}:
                prefix = ""
                encoded.append(self.syllable_vocabulary.unk_id)
                continue
            prefix += token
            encoded.append(self._prefix_id(prefix))
        return encoded

    def _prefix_id(self, prefix: str) -> int:
        if not prefix or len(self.syllable_vocabulary) <= 1:
            return self.syllable_vocabulary.unk_id
        digest = hashlib.sha256(prefix.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % (len(self.syllable_vocabulary) - 1)
        return bucket + 1


def prepare_partial_variant_examples(
    examples: list[dict[str, Any]],
    variant: str,
    character_vocabulary: CharacterVocabulary,
    syllable_vocabulary: SyllableVocabulary,
) -> list[dict[str, Any]]:
    """Prepara E/F usando somente o prefixo silábico observado."""
    if variant not in {"E", "F"}:
        raise ValueError("variant deve ser E ou F")
    encoder = CausalPartialSyllableEncoder(character_vocabulary, syllable_vocabulary)
    prepared = []
    for original in examples:
        example = copy.deepcopy(original)
        context_ids = example.get("context_ids")
        completed_ids = example.get("context_syllable_ids")
        if not isinstance(context_ids, list) or not isinstance(completed_ids, list):
            raise ValueError("exemplo precisa de context_ids e context_syllable_ids")
        if len(context_ids) != len(completed_ids):
            raise ValueError("contextos de caractere e sílaba devem estar alinhados")
        example["context_syllable_ids"] = encoder.encode_context(context_ids, completed_ids)
        if variant == "E":
            vectors = example.get("context_feature_vectors", [])
            example["context_feature_vectors"] = [
                [0.0] * len(vector) for vector in vectors
            ]
        prepared.append(example)
    return prepared
