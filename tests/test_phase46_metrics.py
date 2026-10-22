"""Formal verification of Phase 46 metrics, probe counts, and holdout vocabulary constraints."""

try:
    import pytest
except ModuleNotFoundError:
    import tests._pytest_compat
    import pytest

from kinera.experiments.expanded_pilot import (
    _build_unified_vocabularies,
    compute_probe_partition_breakdown,
)
from kinera.experiments.scientific_metrics import calculate_retention_ratio


def test_positive_retention_ratio_discrete_nature():
    """Verify that R_pos equals k_retained / N_pos for binary probe accuracies."""
    probes = ("probe_malas", "probe_patos", "probe_sapos", "probe_fogos")

    # Case Seed 42 under negative control: N_pos = 4, 2 retained -> R_pos = 2/4 = 0.5000
    ref_seed42 = [1.0, 1.0, 1.0, 1.0]
    post_seed42 = [0.0, 0.0, 1.0, 1.0]
    breakdown_42 = compute_probe_partition_breakdown(probes, ref_seed42, post_seed42)
    assert len(breakdown_42.positive_ref_group_ids) == 4
    k_42 = sum(1 for r in breakdown_42.positive_ref_retention_ratios if r >= 1.0)
    assert k_42 == 2
    assert breakdown_42.mean_positive_ref_retention == pytest.approx(0.5000, abs=1e-5)
    # 2/4 is valid discrete fraction, NOT 1.5/3
    assert k_42 / len(breakdown_42.positive_ref_group_ids) == 0.5000

    # Case Seed 43 under negative control: N_pos = 3, 1 retained -> R_pos = 1/3 = 0.3333
    ref_seed43 = [1.0, 1.0, 1.0, 0.0]
    post_seed43 = [0.0, 0.0, 1.0, 1.0]
    breakdown_43 = compute_probe_partition_breakdown(probes, ref_seed43, post_seed43)
    assert len(breakdown_43.positive_ref_group_ids) == 3
    k_43 = sum(1 for r in breakdown_43.positive_ref_retention_ratios if r >= 1.0)
    assert k_43 == 1
    assert breakdown_43.mean_positive_ref_retention == pytest.approx(1.0 / 3.0, abs=1e-5)

    # Case Seed 44 under negative control: N_pos = 3, 2 retained -> R_pos = 2/3 = 0.6667
    ref_seed44 = [1.0, 1.0, 1.0, 0.0]
    post_seed44 = [1.0, 0.0, 1.0, 1.0]
    breakdown_44 = compute_probe_partition_breakdown(probes, ref_seed44, post_seed44)
    assert len(breakdown_44.positive_ref_group_ids) == 3
    k_44 = sum(1 for r in breakdown_44.positive_ref_retention_ratios if r >= 1.0)
    assert k_44 == 2
    assert breakdown_44.mean_positive_ref_retention == pytest.approx(2.0 / 3.0, abs=1e-5)


def test_independent_holdout_vocabulary_compatibility():
    """Verify that conceptual holdouts with 'i' or 'c' are out of vocabulary,
    explaining why 'pelo' and 'toga' were substituted in the runner."""
    char_vocab, _ = _build_unified_vocabularies()

    # Conceptual words with characters missing in unified vocabulary
    assert "i" not in char_vocab.token_to_id
    assert "c" not in char_vocab.token_to_id

    # 'pipa' and 'toca' contain characters that map to <UNK>
    encoded_pipa = [char_vocab.token_to_id.get(c, char_vocab.token_to_id["<UNK>"]) for c in "pipa"]
    assert char_vocab.token_to_id["<UNK>"] in encoded_pipa

    encoded_toca = [char_vocab.token_to_id.get(c, char_vocab.token_to_id["<UNK>"]) for c in "toca"]
    assert char_vocab.token_to_id["<UNK>"] in encoded_toca

    # Compatible substitute words in the runner must be 100% in-vocabulary
    compatible_words = ["mola", "muda", "muro", "pala", "pelo", "selo", "toga", "vela"]
    for word in compatible_words:
        assert all(c in char_vocab.token_to_id for c in word), f"Word {word} has out-of-vocab characters"
