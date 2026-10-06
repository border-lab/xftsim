"""
Unit tests for the meiosis crossover sampler (reproduce._draw_switches).

The sampler draws switch loci from a Poisson crossover process instead of a
per-locus Bernoulli loop. It must reproduce the per-locus model exactly in
distribution: locus j switches parental copy independently with
probability p[j].
"""
import numpy as np
import pytest

from xftsim.reproduce import (
    _crossover_model, _meiosis_pair_switches_seeded, _meiosis_pair_seeded,
    _switches_to_phase,
)
from xftsim.grg_recombination import _phase_to_segments, _switches_to_segments


P_MIXED = np.array([0.5, 0.0, 0.01, 0.2, 0.45, 0.5, 0.8, 1.0, 0.3, 0.0])


def _switch_draws(p, n_draws):
    model = _crossover_model(p)
    out = np.zeros((n_draws, len(p)), dtype=bool)
    for s in range(n_draws):
        mat, _ = _meiosis_pair_switches_seeded(*model, np.uint32(s))
        out[s, mat] = True
    return out


class TestSwitchDistribution:
    def test_per_locus_frequency_matches_p(self):
        """Each locus switches with probability p[j] (z-test, fixed seeds)."""
        n = 40_000
        freq = _switch_draws(P_MIXED, n).mean(axis=0)
        se = np.sqrt(np.maximum(P_MIXED * (1 - P_MIXED), 1e-12) / n)
        assert np.all(np.abs(freq - P_MIXED) < 5 * se + 1e-12), freq

    def test_deterministic_loci(self):
        """p = 0 never switches; p = 1 always switches."""
        draws = _switch_draws(P_MIXED, 2_000)
        assert not draws[:, P_MIXED == 0.0].any()
        assert draws[:, P_MIXED == 1.0].all()

    def test_loci_switch_independently(self):
        n = 40_000
        draws = _switch_draws(P_MIXED, n)
        joint = np.mean(draws[:, 3] & draws[:, 4])
        expected = P_MIXED[3] * P_MIXED[4]
        se = np.sqrt(expected * (1 - expected) / n)
        assert abs(joint - expected) < 5 * se

    def test_switches_sorted_and_distinct(self):
        model = _crossover_model(np.full(500, 0.3))
        for s in range(50):
            mat, pat = _meiosis_pair_switches_seeded(*model, np.uint32(s))
            for sw in (mat, pat):
                assert np.all(np.diff(sw) > 0)

    def test_phase_wrapper_matches_switches(self):
        p = np.full(60, 0.2)
        p[0] = 0.5
        mat_phase, pat_phase = _meiosis_pair_seeded(p, np.uint32(11))
        mat_sw, pat_sw = _meiosis_pair_switches_seeded(*_crossover_model(p), np.uint32(11))
        np.testing.assert_array_equal(mat_phase, _switches_to_phase(mat_sw, 60))
        np.testing.assert_array_equal(pat_phase, _switches_to_phase(pat_sw, 60))


class TestSamplerRoutes:
    """_crossover_model picks Poisson events for sparse maps and per-locus
    draws for dense ones; both must sample Bernoulli(p) per locus."""

    def test_sparse_map_uses_poisson_route(self):
        p = np.r_[0.5, np.full(999, 1e-4)]
        cumlam, special_idx, _ = _crossover_model(p)
        assert cumlam[-1] > 0
        np.testing.assert_array_equal(special_idx, [0])

    def test_dense_map_uses_per_locus_route(self):
        p = np.r_[0.5, np.full(199, 0.3), 0.0]
        cumlam, special_idx, _ = _crossover_model(p)
        assert cumlam[-1] == 0
        assert special_idx.shape[0] == 200  # every locus with p > 0

    def test_per_locus_route_frequency_matches_p(self):
        p = np.r_[0.5, np.full(199, 0.3), 0.0]
        n = 20_000
        freq = _switch_draws(p, n).mean(axis=0)
        se = np.sqrt(np.maximum(p * (1 - p), 1e-12) / n)
        assert np.all(np.abs(freq - p) < 5 * se + 1e-12)


class TestCrossoverModelValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan])
    def test_rejects_invalid_probabilities(self, bad):
        p = np.array([0.5, 0.1, bad])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _crossover_model(p)


def _phase_to_segments_reference(phase, pos_bp, hap_ids, genome_length):
    """The original per-locus loop, kept as an oracle."""
    m = len(phase)
    if m == 0:
        return [(hap_ids[0], int(genome_length))]
    cur_phase = int(phase[0])
    segments = []
    for j in range(1, m):
        pj = int(phase[j])
        if pj != cur_phase:
            segments.append((hap_ids[cur_phase], int(pos_bp[j])))
            cur_phase = pj
    segments.append((hap_ids[cur_phase], int(genome_length)))
    return segments


class TestSegments:
    def test_phase_to_segments_matches_reference(self):
        rng = np.random.RandomState(0)
        for m in (0, 1, 2, 17, 300):
            pos_bp = np.sort(rng.randint(0, 10_000, size=m))
            for _ in range(20):
                phase = (np.cumsum(rng.rand(m) < 0.1) + rng.randint(2)) % 2
                assert (_phase_to_segments(phase, pos_bp, (7, 9), 10_000)
                        == _phase_to_segments_reference(phase, pos_bp, (7, 9), 10_000))

    def test_switches_to_segments_matches_phase_path(self):
        rng = np.random.RandomState(1)
        m = 200
        pos_bp = np.arange(m) * 50
        model = _crossover_model(np.r_[0.5, np.full(m - 1, 0.05)])
        for s in range(50):
            mat, _ = _meiosis_pair_switches_seeded(*model, np.uint32(s))
            phase = _switches_to_phase(mat, m)
            assert (_switches_to_segments(mat, pos_bp, (3, 4), 99_999)
                    == _phase_to_segments_reference(phase, pos_bp, (3, 4), 99_999))
