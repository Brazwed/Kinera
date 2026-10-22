"""Formal verification of representational identifiability in Regime 1 (unconditioned autoregressive).

Tests the mathematical impossibility of deterministic joint disambiguation under identical
prefix inputs, confirming the exact code origin and bounds of the collision in Kinera.
"""

from __future__ import annotations

import math
try:
    import pytest
except ModuleNotFoundError:
    import tests._pytest_compat
    import pytest

from kinera.language.orthography.features import annotate_record
from kinera.language.datasets.builder import build_autoregressive_examples
from kinera.language.vocabulary import CharacterVocabulary, SyllableVocabulary
from kinera.nn.models.hybrid import AutoregressiveModel


from kinera.experiments.expanded_pilot import _build_unified_vocabularies


def _build_test_vocabularies() -> tuple[CharacterVocabulary, SyllableVocabulary]:
    return _build_unified_vocabularies()


WORD_SYLLABLES: dict[str, list[str]] = {
    "mala": ["ma", "la"],
    "medo": ["me", "do"],
    "pato": ["pa", "to"],
    "povo": ["po", "vo"],
}


def test_regime1_input_fields_are_strictly_identical():
    """Verify that all input fields at transition [1] are identical for conflicting pairs."""
    char_vocab, syl_vocab = _build_test_vocabularies()

    rec_mala = annotate_record({"word_key": "mala", "word_normalized": "mala", "syllables": WORD_SYLLABLES["mala"]})
    rec_medo = annotate_record({"word_key": "medo", "word_normalized": "medo", "syllables": WORD_SYLLABLES["medo"]})

    exs_mala = build_autoregressive_examples([rec_mala], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)
    exs_medo = build_autoregressive_examples([rec_medo], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)

    # Transition [1] corresponds to target_position = 3, char_index = 1
    t1_mala = exs_mala[1]
    t1_medo = exs_medo[1]

    # Context character tokens: ['<START>', 'm']
    assert t1_mala["context_ids"] == t1_medo["context_ids"]
    assert [char_vocab.id_to_token[i] for i in t1_mala["context_ids"]] == ["<START>", "m"]

    # Context syllable tokens: ['<UNK_SYL>', '<UNK_SYL>'] (causally masked)
    assert t1_mala["context_syllable_ids"] == t1_medo["context_syllable_ids"]
    assert [syl_vocab.id_to_token[i] for i in t1_mala["context_syllable_ids"]] == ["<UNK_SYL>", "<UNK_SYL>"]

    # Context feature vectors: [[0.0]*10, features('m')]
    assert t1_mala["context_feature_vectors"] == t1_medo["context_feature_vectors"]

    # But targets are conflicting and mutually exclusive!
    assert t1_mala["target_char"] == "a"
    assert t1_medo["target_char"] == "e"
    assert t1_mala["target_char_id"] != t1_medo["target_char_id"]

    # Same check for pato vs povo
    rec_pato = annotate_record({"word_key": "pato", "word_normalized": "pato", "syllables": WORD_SYLLABLES["pato"]})
    rec_povo = annotate_record({"word_key": "povo", "word_normalized": "povo", "syllables": WORD_SYLLABLES["povo"]})

    exs_pato = build_autoregressive_examples([rec_pato], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)
    exs_povo = build_autoregressive_examples([rec_povo], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)

    t1_pato = exs_pato[1]
    t1_povo = exs_povo[1]

    assert t1_pato["context_ids"] == t1_povo["context_ids"]
    assert [char_vocab.id_to_token[i] for i in t1_pato["context_ids"]] == ["<START>", "p"]
    assert t1_pato["context_syllable_ids"] == t1_povo["context_syllable_ids"]
    assert t1_pato["context_feature_vectors"] == t1_povo["context_feature_vectors"]

    assert t1_pato["target_char"] == "a"
    assert t1_povo["target_char"] == "o"
    assert t1_pato["target_char_id"] != t1_povo["target_char_id"]


@pytest.mark.parametrize("seed", [42, 43, 44, 999])
def test_regime1_representations_and_logits_are_mathematically_identical(seed: int):
    """Verify that any deterministic initialization yields zero vector distance and identical argmax."""
    char_vocab, syl_vocab = _build_test_vocabularies()
    model = AutoregressiveModel(len(char_vocab), len(syl_vocab), seed=seed)

    rec_mala = annotate_record({"word_key": "mala", "word_normalized": "mala", "syllables": WORD_SYLLABLES["mala"]})
    rec_medo = annotate_record({"word_key": "medo", "word_normalized": "medo", "syllables": WORD_SYLLABLES["medo"]})

    t1_mala = build_autoregressive_examples([rec_mala], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)[1]
    t1_medo = build_autoregressive_examples([rec_medo], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)[1]

    logits_mala = model.forward(t1_mala["context_ids"], t1_mala["context_syllable_ids"], t1_mala["context_feature_vectors"])
    pooled_mala = list(model.last_pooled)

    logits_medo = model.forward(t1_medo["context_ids"], t1_medo["context_syllable_ids"], t1_medo["context_feature_vectors"])
    pooled_medo = list(model.last_pooled)

    # Euclidean distance between pooled representations: exactly 0.0
    euclid_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pooled_mala, pooled_medo)))
    assert euclid_dist == 0.0

    # Cosine distance: less than machine epsilon
    dot = sum(a * b for a, b in zip(pooled_mala, pooled_medo))
    norm_a = math.sqrt(sum(a ** 2 for a in pooled_mala))
    norm_b = math.sqrt(sum(b ** 2 for b in pooled_medo))
    cos_dist = 1.0 - (dot / (norm_a * norm_b))
    assert abs(cos_dist) < 1e-14

    # Logits distance: exactly 0.0
    euclid_logits = math.sqrt(sum((a - b) ** 2 for a, b in zip(logits_mala, logits_medo)))
    assert euclid_logits == 0.0

    # Deterministic argmax output: must be identical
    pred_mala = max(range(len(logits_mala)), key=logits_mala.__getitem__)
    pred_medo = max(range(len(logits_medo)), key=logits_medo.__getitem__)
    assert pred_mala == pred_medo

    # Proves mathematical impossibility of joint correctness:
    # A single integer prediction cannot simultaneously equal two different target integers
    target_mala = t1_mala["target_char_id"]
    target_medo = t1_medo["target_char_id"]
    correct_mala = int(pred_mala == target_mala)
    correct_medo = int(pred_medo == target_medo)
    assert correct_mala + correct_medo <= 1


def test_causal_masking_prevents_future_vowel_leakage():
    """Verify that builder strictly masks syllable IDs until the syllable boundary is observed."""
    char_vocab, syl_vocab = _build_test_vocabularies()
    rec_mala = annotate_record({"word_key": "mala", "word_normalized": "mala", "syllables": WORD_SYLLABLES["mala"]})
    exs = build_autoregressive_examples([rec_mala], char_vocab, context_size=2, syllable_vocabulary=syl_vocab)

    # Step 0: ['<START>', '<START>'], target 'm' -> syllables unmasked: None
    assert exs[0]["context_syllable_ids"] == [syl_vocab.unk_id, syl_vocab.unk_id]

    # Step 1: ['<START>', 'm'], target 'a' -> 'm' is not end of syllable 'ma'.
    # Syllable 'ma' must NOT be revealed yet (which would leak target vowel 'a')
    assert exs[1]["context_syllable_ids"] == [syl_vocab.unk_id, syl_vocab.unk_id]

    # Step 2: ['m', 'a'], target 'l' -> 'a' completed syllable 'ma'.
    # Now and only now, syllable 'ma' is legitimately in the completed history!
    assert exs[2]["context_syllable_ids"] == [syl_vocab.encode("ma"), syl_vocab.encode("ma")]
