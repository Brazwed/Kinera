"""Deterministic character vocabulary with stable special tokens."""

from __future__ import annotations

from typing import Any

SPECIAL_TOKENS = ("<START>", "<END>", "<UNK>")


class CharacterVocabulary:
    """Map Unicode characters and special tokens to stable integer IDs."""

    def __init__(self, tokens: list[str]) -> None:
        if not tokens or tuple(tokens[:3]) != SPECIAL_TOKENS:
            raise ValueError("os três primeiros tokens devem ser os especiais")
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulário não pode conter tokens duplicados")
        self.tokens = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.id_to_token = {index: token for index, token in enumerate(tokens)}
        self.start_id = self.token_to_id["<START>"]
        self.end_id = self.token_to_id["<END>"]
        self.unk_id = self.token_to_id["<UNK>"]

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "CharacterVocabulary":
        characters = set()
        for record in records:
            word = record.get("word_normalized", record.get("word_raw", ""))
            if not isinstance(word, str):
                raise TypeError("word do registro deve ser string")
            characters.update(word)
        return cls([str(token) for token in SPECIAL_TOKENS] + sorted(characters))

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, *, add_boundaries: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text deve ser uma string")
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_boundaries:
            return [self.start_id, *ids, self.end_id]
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = False) -> str:
        if not isinstance(ids, list):
            raise TypeError("ids deve ser uma lista")
        tokens = []
        for index in ids:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("cada ID deve ser inteiro")
            if index not in self.id_to_token:
                raise ValueError(f"ID fora do vocabulário: {index}")
            token = self.id_to_token[index]
            if skip_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return "".join(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "tokens": self.tokens.copy()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterVocabulary":
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("formato de vocabulário inválido")
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
            raise ValueError("tokens inválidos")
        return cls(tokens)
