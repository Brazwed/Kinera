"""Stable, type-preserving hashes for scientific experiment inputs."""

from __future__ import annotations

import hashlib
import math
from typing import Any


def _write_length(digest: Any, length: int) -> None:
    digest.update(length.to_bytes(8, byteorder="big", signed=False))


def _update_canonical(digest: Any, value: Any) -> None:
    """Stream a typed canonical representation into ``digest``.

    Supported values mirror JSON data used by the experiment runner. Mapping
    keys are strings and sorted by Unicode code point; list order is retained.
    Floats use their exact hexadecimal representation, not locale or repr().
    """
    value_type = type(value)
    if value is None:
        digest.update(b"n")
    elif value_type is bool:
        digest.update(b"b1" if value else b"b0")
    elif value_type is int:
        payload = str(value).encode("ascii")
        digest.update(b"i")
        _write_length(digest, len(payload))
        digest.update(payload)
    elif value_type is float:
        if not math.isfinite(value):
            raise ValueError("fingerprint não aceita float NaN ou infinito")
        payload = value.hex().encode("ascii")
        digest.update(b"f")
        _write_length(digest, len(payload))
        digest.update(payload)
    elif value_type is str:
        payload = value.encode("utf-8")
        digest.update(b"s")
        _write_length(digest, len(payload))
        digest.update(payload)
    elif value_type is list:
        digest.update(b"l")
        _write_length(digest, len(value))
        for item in value:
            _update_canonical(digest, item)
    elif value_type is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("fingerprint exige chaves de dicionário string")
        keys = sorted(value)
        digest.update(b"d")
        _write_length(digest, len(keys))
        for key in keys:
            _update_canonical(digest, key)
            _update_canonical(digest, value[key])
    else:
        raise TypeError(
            f"tipo não suportado no fingerprint: {value_type.__name__}"
        )


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 of a deterministic, typed encoding of JSON-like data."""
    digest = hashlib.sha256()
    _update_canonical(digest, value)
    return digest.hexdigest()
