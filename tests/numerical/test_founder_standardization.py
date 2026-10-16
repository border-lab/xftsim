"""
Genetic values across generations: standardize_with='founders' vs 'current'.

Spec (devtools/math_spec.md, "Across generations"): with
standardize_with='founders', generation t's genetic value is
    G_t = ((X_t - 2 p0) / sqrt(2 p0 q0)) @ beta,
with p0 the generation-0 empirical allele frequencies, so each allele's
effect beta_j / sqrt(2 p0 q0) is fixed. With 'current', p0 is replaced by
generation t's own frequencies p_t. Expected values below are computed
directly from genotypes with numpy, not through xftsim.
"""
import json
import os

import numpy as np
import pytest

import xftsim as xft
from xftsim.arch import (Architecture, GeneticComponent, MVGeneticComponent,
                         NoiseComponent, AggregationComponent)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta


def _rare_variant_founders(n, m, seed):
    """Founders whose allele frequencies are skewed toward rare variants."""
    rng = np.random.RandomState(seed)
    af = np.clip(rng.beta(0.3, 3.0, size=m), 1.0 / (2 * n), 0.5)
    geno = (rng.rand(n, m, 2) < af[None, :, None]).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=SampleMeta(iid=np.arange(n)),
                               variants=VariantMeta(vid=np.arange(m), af=af))


def _sim(hap, mode, n_gen, effects, seed=1, **kw):
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    sim = Simulation(hap, arch, RandomMating(),
                     RecombinationMap.constant_map(m=hap.m, p=0.5), seed=seed,
                     retain_haplotypes=n_gen, retain_phenotypes=n_gen,
                     standardize_with=mode, **kw)
    sim.run(n_gen)
    return sim


def _standardized_value(X, p, beta):
    """Spec formula: ((X - 2p) / sqrt(2pq)) @ beta, with sqrt(2pq)=0 -> 1."""
    denom = np.sqrt(2 * p * (1 - p))
    denom[denom == 0] = 1.0
    return ((X[:, :, 0] + X[:, :, 1] - 2 * p) / denom) @ beta


N, M, T = 300, 400, 6


@pytest.fixture(scope="module")
def runs():
    hap = _rare_variant_founders(N, M, seed=3)
    eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=4)
    return (hap, eff, _sim(hap, 'founders', T, eff), _sim(hap, 'current', T, eff))


class TestSpecFormula:
    def test_founders_mode_fixes_allele_effects(self, runs):
        hap, eff, founders, _ = runs
        p0 = hap.genotypes.mean(axis=(0, 2))
        for g in range(T):
            X = founders.haplotype_history[g].genotypes
            np.testing.assert_allclose(founders.phenotype_history[g]['Y.G'],
                                       _standardized_value(X, p0, eff.effects),
                                       atol=1e-10, err_msg=f"generation {g}")

    def test_current_mode_restandardizes(self, runs):
        _, eff, _, current = runs
        for g in range(T):
            X = current.haplotype_history[g].genotypes
            p_t = X.mean(axis=(0, 2))
            np.testing.assert_allclose(current.phenotype_history[g]['Y.G'],
                                       _standardized_value(X, p_t, eff.effects),
                                       atol=1e-10, err_msg=f"generation {g}")

    def test_founder_af_is_generation0_empirical(self, runs):
        """Not variants.af, which here holds the frequencies sampled from."""
        hap, _, founders, _ = runs
        np.testing.assert_allclose(founders.founder_af,
                                   hap.genotypes.mean(axis=(0, 2)), rtol=1e-12)
        assert not np.allclose(founders.founder_af, hap.variants.af)

    def test_generation0_and_genotypes_identical_across_modes(self, runs):
        _, _, founders, current = runs
        np.testing.assert_array_equal(founders.phenotype_history[0]['Y'],
                                      current.phenotype_history[0]['Y'])
        for g in range(T):
            np.testing.assert_array_equal(founders.haplotype_history[g].genotypes,
                                          current.haplotype_history[g].genotypes)

    def test_multivariate_component(self):
        hap = _rare_variant_founders(200, 100, seed=5)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=100, seed=6)
        arch = Architecture()
        arch.add(['A.G', 'B.G'], MVGeneticComponent(eff))
        sim = Simulation(hap, arch, RandomMating(),
                         RecombinationMap.constant_map(m=100, p=0.5), seed=1,
                         retain_haplotypes=4, retain_phenotypes=4)
        sim.run(4)
        p0 = hap.genotypes.mean(axis=(0, 2))
        X = sim.haplotype_history[3].genotypes
        expected = _standardized_value(X, p0, eff.effects)
        np.testing.assert_allclose(sim.phenotype_history[3]['A.G'], expected[:, 0], atol=1e-10)
        np.testing.assert_allclose(sim.phenotype_history[3]['B.G'], expected[:, 1], atol=1e-10)

    def test_raw_effects_unaffected(self):
        hap = _rare_variant_founders(200, 100, seed=7)
        eff = AdditiveEffects.from_array(np.linspace(-0.1, 0.1, 100), standardized=False)
        a = _sim(hap, 'founders', 4, eff)
        b = _sim(hap, 'current', 4, eff)
        for g in range(4):
            np.testing.assert_array_equal(a.phenotype_history[g]['Y.G'],
                                          b.phenotype_history[g]['Y.G'])

    def test_architecture_compute_alone_uses_current_frequencies(self):
        hap = _rare_variant_founders(100, 50, seed=8)
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=9)
        arch = Architecture()
        arch.add('G', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=np.random.RandomState(0))
        np.testing.assert_allclose(
            pheno['G'], _standardized_value(hap.genotypes, hap.af_empirical, eff.effects),
            atol=1e-10)

    def test_invalid_mode(self):
        hap = _rare_variant_founders(20, 10, seed=1)
        with pytest.raises(ValueError, match="standardize_with"):
            Simulation(hap, Architecture(), RandomMating(),
                       RecombinationMap.constant_map(m=10, p=0.5),
                       standardize_with='founder')


class TestVarianceRatchet:
    def test_founders_mode_has_no_ratchet(self):
        """
        Spec ("Across generations"): with fixed allele effects
        a_j = beta_j / sqrt(2 p0 q0), the genic variance is
        sum_j a_j^2 2 p_t q_t, which under random mating decays like
        heterozygosity, (1 - 1/(2 N_e))^t with N_e ~ 2N for two offspring
        per individual. Re-standardizing ('current') instead gives
        sum over still-segregating sites of beta_j^2, which falls as rare
        variants are lost. Both expectations are computed from allele
        frequencies; realized var(G) adds LD sampling noise of roughly
        sqrt(2/N) = 10% per generation (about 7% across seeds after
        averaging the last six generations, so +-0.25 is about 3.5 sd).
        Calibrated over 10 seeds: var/genic 0.85-1.10 in both modes,
        founders genic ratio 0.90-1.04, current/founders ratio <= 0.83.
        """
        n, m, n_gen = 200, 2000, 16
        hap = _rare_variant_founders(n, m, seed=100)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=0)
        founders = _sim(hap, 'founders', n_gen, eff, seed=0)
        current = _sim(hap, 'current', n_gen, eff, seed=0)
        late = slice(n_gen - 6, n_gen)

        beta = eff.effects
        p0 = hap.genotypes.mean(axis=(0, 2))
        d0 = np.sqrt(2 * p0 * (1 - p0))
        d0[d0 == 0] = 1.0
        a = beta / d0
        # Genotypes are identical in both modes under random mating.
        P = np.array([founders.haplotype_history[g].genotypes.mean(axis=(0, 2))
                      for g in range(n_gen)])
        genic_founders = (a ** 2 * 2 * P * (1 - P)).sum(axis=1)
        genic_current = np.array([np.sum(beta[(p > 0) & (p < 1)] ** 2) for p in P])

        def realized(sim):
            return np.array([np.var(sim.phenotype_history[g]['Y.G'])
                             for g in range(n_gen)])

        # Each mode's genetic variance matches its own spec expectation.
        assert abs(realized(founders)[late].mean() / genic_founders[late].mean() - 1) < 0.25
        assert abs(realized(current)[late].mean() / genic_current[late].mean() - 1) < 0.25

        # Fixed effects: genic variance follows heterozygosity decay.
        theory = (1 - 1 / (4 * n)) ** (n_gen - 3.5)
        founders_ratio = genic_founders[late].mean() / genic_founders[0]
        assert abs(founders_ratio - theory) < 0.15

        # Re-standardizing ratchets it down as rare variants are lost.
        current_ratio = genic_current[late].mean() / genic_current[0]
        assert current_ratio < 0.9 * founders_ratio


class TestCheckpoint:
    def _run(self, mode, n_gen, retain_haplotypes=1):
        hap = _rare_variant_founders(N, M, seed=3)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=4)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        return Simulation(hap, arch, RandomMating(),
                          RecombinationMap.constant_map(m=M, p=0.5), seed=1,
                          standardize_with=mode, retain_haplotypes=retain_haplotypes)

    def test_resume_matches_uninterrupted(self, tmp_path):
        full = self._run('founders', 6)
        full.run(6)
        part = self._run('founders', 3)
        part.run(3)
        assert 0 not in part.haplotype_history  # generation 0 already pruned
        xft.io.save_simulation_checkpoint(part, str(tmp_path))
        resumed = Simulation.from_checkpoint(str(tmp_path))
        assert resumed.standardize_with == 'founders'
        np.testing.assert_array_equal(resumed.founder_af, full.founder_af)
        resumed.continue_run(3)
        np.testing.assert_array_equal(resumed.phenotypes['Y.G'], full.phenotypes['Y.G'])

    def _make_old_checkpoint(self, path, **kw):
        """A 'current'-mode checkpoint in the pre-option on-disk format."""
        part = self._run('current', 3, **kw)
        part.run(3)
        xft.io.save_simulation_checkpoint(part, path)
        os.remove(os.path.join(path, 'founder_af.npz'))
        meta_path = os.path.join(path, 'meta.json')
        meta = json.load(open(meta_path))
        del meta['standardize_with']
        json.dump(meta, open(meta_path, 'w'))

    def test_old_checkpoint_resumes_as_current(self, tmp_path):
        self._make_old_checkpoint(str(tmp_path))
        with pytest.warns(UserWarning, match="predates"):
            resumed = Simulation.from_checkpoint(str(tmp_path))
        assert resumed.standardize_with == 'current'
        resumed.continue_run(3)
        full = self._run('current', 6)
        full.run(6)
        np.testing.assert_array_equal(resumed.phenotypes['Y.G'], full.phenotypes['Y.G'])

    def test_old_checkpoint_cannot_switch_without_founders(self, tmp_path):
        self._make_old_checkpoint(str(tmp_path))
        with pytest.raises(ValueError, match="founders' allele"):
            Simulation.from_checkpoint(str(tmp_path), standardize_with='founders')

    def test_old_dense_checkpoint_with_generation0_can_switch(self, tmp_path):
        self._make_old_checkpoint(str(tmp_path), retain_haplotypes=5)
        resumed = Simulation.from_checkpoint(str(tmp_path), standardize_with='founders')
        full = self._run('founders', 3)
        full.run(3)
        np.testing.assert_array_equal(resumed.founder_af, full.founder_af)

    def test_continue_without_founder_frequencies_raises(self):
        sim = self._run('founders', 3)
        sim._mate_assignments[0] = RandomMating().mate(
            sim.haplotypes.samples, rng=np.random.RandomState(0))
        with pytest.raises(RuntimeError, match="founders' allele"):
            sim.continue_run(1)

    def test_switch_mode_on_resume(self, tmp_path):
        part = self._run('founders', 3)
        part.run(3)
        xft.io.save_simulation_checkpoint(part, str(tmp_path))
        resumed = Simulation.from_checkpoint(str(tmp_path), standardize_with='current')
        assert resumed.standardize_with == 'current'
        assert resumed._reference_af() is None

    def test_wrong_length_founder_af(self, tmp_path):
        part = self._run('founders', 2)
        part.run(2)
        xft.io.save_simulation_checkpoint(part, str(tmp_path))
        np.savez(os.path.join(str(tmp_path), 'founder_af.npz'), founder_af=np.zeros(3))
        with pytest.raises(ValueError, match="shape"):
            Simulation.from_checkpoint(str(tmp_path))
