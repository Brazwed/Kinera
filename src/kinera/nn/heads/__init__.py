"""Output heads for character, syllable, and boundary prediction."""

from .boundary import BoundaryHead
from .character import CharacterHead
from .multitask import MultiTaskHeads
from .syllable import SyllableHead

__all__ = ["BoundaryHead", "CharacterHead", "MultiTaskHeads", "SyllableHead"]
