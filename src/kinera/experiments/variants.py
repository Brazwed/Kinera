"""Named variant sets and validation for controlled experiments."""

from __future__ import annotations

SUPPORTED_VARIANTS = ("A", "B", "C", "D")
CONTROLLED_VARIANTS = SUPPORTED_VARIANTS
DEFAULT_VARIANTS = ("A", "C", "D")
BASELINE_VARIANTS = ("A",)
PRIMARY_VARIANTS = ("C", "D")
OPTIONAL_ABLATION_VARIANTS = ("B",)
# Compatibility alias; new callers should name the intended set.
VARIANTS = CONTROLLED_VARIANTS


def validate_variant_selection(variants: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(variants, str):
        raise TypeError("variants deve ser uma sequência de variantes")
    try:
        selected = tuple(variants)
    except TypeError as error:
        raise TypeError("variants deve ser uma sequência de variantes") from error
    if not selected:
        raise ValueError("variants não pode ser vazio")
    for variant in selected:
        if not isinstance(variant, str) or variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variante desconhecida: {variant}")
    if len(set(selected)) != len(selected):
        raise ValueError("variants não pode conter duplicatas")
    return selected


_validate_variant_selection = validate_variant_selection

__all__ = [
    "BASELINE_VARIANTS",
    "CONTROLLED_VARIANTS",
    "DEFAULT_VARIANTS",
    "OPTIONAL_ABLATION_VARIANTS",
    "PRIMARY_VARIANTS",
    "SUPPORTED_VARIANTS",
    "VARIANTS",
    "validate_variant_selection",
]
