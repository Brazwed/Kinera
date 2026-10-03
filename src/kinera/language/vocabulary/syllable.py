"""Deterministic vocabulary for observed syllables."""

from __future__ import annotations

from typing import Any

SYLLABLE_UNKNOWN_TOKEN = "<UNK_SYL>"


class SyllableVocabulary:
    """Map observed syllables to IDs with an explicit unknown token."""

    def __init__(self, tokens: list[str]) -> None:
        if not tokens or tokens[0] != SYLLABLE_UNKNOWN_TOKEN:
            raise ValueError("o primeiro token deve ser <UNK_SYL>")
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulário não pode conter tokens duplicados")
        self.tokens = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = {index: token for index, token in enumerate(tokens)}
        self.unk_id = self.token_to_id[SYLLABLE_UNKNOWN_TOKEN]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "SyllableVocabulary":
        syllables: set[str] = set()
        for record in records:
            values = record.get("syllables", [])
            if not isinstance(values, list):
                raise TypeError("syllables do registro deve ser uma lista")
            for syllable in values:
                if not isinstance(syllable, str) or not syllable:
                    raise ValueError("cada sílaba deve ser uma string não vazia")
                syllables.add(syllable)
        return cls([SYLLABLE_UNKNOWN_TOKEN, *sorted(syllables)])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, syllable: str) -> int:
        if not isinstance(syllable, str):
            raise TypeError("syllable deve ser uma string")
        return self.token_to_id.get(syllable, self.unk_id)

    def decode(self, index: int) -> str:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index deve ser inteiro")
        if index not in self.id_to_token:
            raise ValueError(f"ID fora do vocabulário: {index}")
        return self.id_to_token[index]

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "tokens": self.tokens.copy()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyllableVocabulary":
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("formato de vocabulário silábico inválido")
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
            raise ValueError("tokens silábicos inválidos")
        return cls(tokens)
